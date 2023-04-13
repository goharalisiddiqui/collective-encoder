
import numpy as np
import copy
import os
import warnings

from scipy import stats as scipy_stats

class SchedulerBetaParent:
    def __init__(self, bresetopt, outputpath, a_init, angular, a_end=1.0, intwidth=None,
                 convergence_crit_type='check_loss_increase'):

        self.convergence_crit_type = convergence_crit_type
        self.tempering_training_iteration = 0
        self.usesched = True
        self.bresetOptimizer = False
        self.bActiveResetOptimizer = bresetopt
        self.steps_enforce_step = 200 # 300#1500 # 1000
        self.sel_conv_criterium = 'decay'# or interval
        self.angular = angular

        #self.tempering_training_itermax = 100000
        #self.tempering_check_change_every = 200
        self.tempering_check_change_every = 200
        self.tempering_file = os.path.join(outputpath, 'a_prefactor.txt')
        self.interval_file = os.path.join(outputpath, 'conv_interval.txt')
        self.step_history_file = os.path.join(outputpath, 'stepper_history')
        self.step_conviter_file = os.path.join(outputpath, 'conviter_history.txt')
        self.step_history = []
        self.list_conv_steps = []
        self.converged_steps = 0
        self.loss = []
        self.lastconvergediteration = []
        self.lastupdateatiter = 0
        self.a = a_init
        self.a_init = a_init
        self.a_end = a_end
        self.a_inactive_sched = 1.0
        self.assess_loss = None
        self.conv_attempts = 0
        self.count_too_small_f = 0

        if not intwidth:
            # this was used originally
            convCheckOptionsGauss = {'miniterations': 400, 'intvervalwidth': 0.005, 'initintvervalwidth': 0.005,
                                     'miniterationsperconv': 300}
            # for convergence checking actual interval
            convCheckOptionsGauss = {'miniterations': 250, 'intvervalwidth': 0.07, 'initintvervalwidth': 0.07,
                                     'miniterationsperconv': 100}
            # for convergence checking only lower component
            convCheckOptionsGauss = {'miniterations': 400, 'intvervalwidth': 0.005, 'initintvervalwidth': 0.005,
                                     'miniterationsperconv': 300}
        else:
            convCheckOptionsGauss = {'miniterations': 400, 'intvervalwidth': intwidth, 'initintvervalwidth': intwidth,
                                     'miniterationsperconv': 300}
            convCheckOptionsGauss = {'miniterations': 800, 'intvervalwidth': intwidth, 'initintvervalwidth': intwidth,
                                 'miniterationsperconv': 100}

        self.convCheckOptions = convCheckOptionsGauss

    def getconvergenceinterval(self, last_loss):
        interval = self.convCheckOptions['intvervalwidth'] * last_loss
        interval = 1000 if interval > 1000 else interval
        return interval

    def set_loss_evaluation_function(self, assess_loss):
        self.assess_loss = assess_loss

    def getLearningPrefactor(self):
        if self.usesched:
            return self.a
        else:
            return self.a_inactive_sched

    def doStep(self):
        self.tempering_training_iteration += 1

    def setSched(self, bschedactive, set_fixed_individual_beta=None):
        self.usesched = bschedactive
        if set_fixed_individual_beta is not None:
            self.a_inactive_sched = set_fixed_individual_beta

    def resetoptimizer(self):
        if self.bActiveResetOptimizer:
            if self.bresetOptimizer:
                self.bresetOptimizer = False
                return True
            else:
                return False
        else:
            return False

    def getLastUpdateIteration(self):
        # check if list is not empty
        if self.step_history:
            return self.step_history[-1]['training_iteration']
        else:
            return 0.

    def writeHistory(self, rel_kl_increase=None, incline=None):
        self.list_conv_steps.append(self.converged_steps)
        if rel_kl_increase is None:
            self.step_history.append({'converged_step': self.converged_steps, 'a_val': self.a,
                                  'training_iteration': self.tempering_training_iteration})
        else:
            self.step_history.append({'converged_step': self.converged_steps, 'a_val': self.a,
                                      'training_iteration': self.tempering_training_iteration,
                                      'rel_kl_inc': rel_kl_increase.item(), 'grad_incline': incline})
        np.save(self.step_history_file, self.step_history)
        np.savetxt(self.step_conviter_file, self.list_conv_steps)

    def updateLoss(self, loss):
        self.loss = copy.copy(loss)

    def enforceUpdateStep(self):
        return self.steps_enforce_step < (self.tempering_training_iteration - self.getLastUpdateIteration())

    def criterium_interval(self, iteration, batched_loss):

        if os.path.isfile(self.interval_file):
            try:
                interval = np.loadtxt(self.interval_file)
            except:
                raise ValueError('Loading interval file failed.')
        else:
            if iteration > self.convCheckOptions['miniterations']:
                interval = self.getconvergenceinterval(batched_loss[-1])
            else:
                if self.angular:
                    interval = 0.02 * batched_loss[-1]
                else:
                    interval = 1.e-20

        # this is interval based
        below_upper = [batched_loss[-m] < (batched_loss[-1] + interval) for m in range(2, self.nlast + 1)]
        above_lower = [batched_loss[-m] > (batched_loss[-1] - interval) for m in range(2, self.nlast + 1)]
        if all(below_upper) and all(above_lower):  # and self.checkMinIterations():
            self.lastconvergediteration.append(iteration)
            return True
        else:
            return False

    def criterium_decay(self, batched_loss):
        # this is based if the loss keeps decaying
        len_batched_loss = len(batched_loss)
        min_batched_loss = min(batched_loss)
        min_index = np.argmin(batched_loss)

        above_min = [batched_loss[-m] >= min_batched_loss for m in range(1, self.nlast + 1)]
        threshold_above_min = [batched_loss[-m] < 1.5 * min_batched_loss for m in range(1, self.nlast + 1)]
        # the right expression checks that the minimum is not at the very last position.

        if all(above_min) and len_batched_loss >= min_index + 2:
            self.conv_attempts = 0
            return True
        #if all(above_min) and all(threshold_above_min) and len_batched_loss >= min_index + 2:
        #    self.conv_attempts = 0
        #    return True
        #elif self.conv_attempts > 4 and all(above_min) and len_batched_loss >= min_index + 2:
        #    self.conv_attempts = 0
        #    return True
        else:
            self.conv_attempts += 1
            return False

    def check_conv(self):
        iteration = len(self.loss)
        iteration_last_update = int(self.getLastUpdateIteration())
        # only check if we sufficiently long run with the changed a value.
        iterations_current_beta = iteration - iteration_last_update
        min_iter = self.convCheckOptions['miniterations']

        # note: for siam the following two lines were off
        #if self.a < 1e-8:
        #    min_iter *= 0.5

        # note: for siam - min_iter was not checked!
        if iterations_current_beta > (self.obsinterval + 1) * (self.nlast + 2):
        #if iterations_current_beta > (self.obsinterval+1)*(self.nlast+2) and iterations_current_beta > min_iter:
            loss_list = self.loss[iteration_last_update:]
            batches = int((iterations_current_beta) / self.obsinterval)
            minoflist = min(loss_list)
            loss_list = [loss_list[m] - minoflist + 1 for m in range(0, iterations_current_beta)]
            #loss_list = [loss_list[m] - 0. + 1 for m in range(iteration)]
            batched_loss = [np.mean(loss_list[m*self.obsinterval:(m+1)*self.obsinterval]) for m in range(batches)]

            if self.sel_conv_criterium is 'decay':
                bconverged = self.criterium_decay(batched_loss)
            elif self.sel_conv_criterium is 'interval':
                bconverged = self.criterium_interval(iteration, batched_loss)
            else:
                raise ValueError('Convergence criterium {} not implemented'.format(self.sel_conv_criterium))
            return bconverged
        else:
            return False

    def check_conv_grad(self, gradient_norm_array, window_size=100):
        iteration = len(self.loss)
        iteration_last_update = int(self.getLastUpdateIteration())
        gradient_norm_decay = 2.
        if iteration - iteration_last_update > window_size:
            if gradient_norm_array is None:
                raise ValueError('Gradient norm is not defined.')

            # use last window to check convergence
            grad_norm_current_step = gradient_norm_array[-window_size:]
            x = np.arange(len(grad_norm_current_step))
            gradient, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, grad_norm_current_step)
            if np.abs(gradient) < gradient_norm_decay:
                return True, gradient
            else:
                return False, 100.
        else:
            return False, 100.

    def checkConvBoth(self):
        iteration = len(self.loss)
        iteration_last_update = int(self.getLastUpdateIteration())
        iteration_since_last_update = iteration - iteration_last_update
        # only check if we sufficiently long run with the changed a value.
        if iteration_since_last_update > (self.obsinterval + 1) * (self.nlast + 2) and iteration_since_last_update > self.convCheckOptions['miniterations']:
            # loss_list = self.loss
            loss_list = self.loss[iteration_last_update:]
            batches = int((iteration - iteration_last_update) / self.obsinterval)
            minoflist = min(loss_list)
            loss_list = [loss_list[m] - minoflist + 1 for m in range(0, iteration - iteration_last_update)]
            # loss_list = [loss_list[m] - 0. + 1 for m in range(iteration)]
            batched_loss = [np.mean(loss_list[m*self.obsinterval:(m+1)*self.obsinterval]) for m in range(batches)]

            if os.path.isfile(self.interval_file):
                try:
                    interval = np.loadtxt(self.interval_file)
                except:
                    raise ValueError('Loading interval file failed.')
            else:
                if iteration_since_last_update > self.convCheckOptions['miniterations']:
                    interval = self.getconvergenceinterval(batched_loss[-1])
                else:
                    if self.angular:
                        interval = 0.02 * batched_loss[-1]
                    else:
                        interval = 1.e-20

            below_upper = [batched_loss[-m] < (batched_loss[-1] + interval) for m in range(2, self.nlast+1)]
            above_lower = [batched_loss[-m] > (batched_loss[-1] - interval) for m in range(2, self.nlast+1)]

            if all(below_upper) and all(above_lower): #and self.checkMinIterations():
                self.lastconvergediteration.append(iteration)
                return True
            else:
                return False
        else:
            return False

class SchedulerBetaKL(SchedulerBetaParent):
    def __init__(self, a_init, outputpath, bresetopt=False, angular=False, a_end=1.0, intwidth=None,
                 max_beta_increase=1.e-3, eval_method_kl='Standard', convergence_crit_type='check_loss_increase', max_kl_inc=.75):
        super(SchedulerBetaKL, self).__init__(bresetopt=bresetopt, outputpath=outputpath, a_init=a_init,
                                              angular=angular, intwidth=intwidth,
                                              convergence_crit_type=convergence_crit_type)
        self.nlast = 3
        self.obsinterval = 5 #20
        # 2019-05-02
        # self.obsinterval = 5

        self.relmax_kl_increase = max_kl_inc #.75#.01
        self.max_beta_increase = max_beta_increase
        #self.absmax_beta_increase = 1.e-3
        self.eval_method_kl = eval_method_kl

    #TODO clean importance_sampler overload - it is not required here
    def updateLearningPrefactor(self, kl_current, importance_sampler=None, gradient_norm=None):
        incline = None
        if self.usesched:
            if self.a >= 1.0:
                self.a = 1.0
            # aviod evaluation every iteration
            elif not (self.tempering_training_iteration % self.obsinterval):
                if not self.tempering_training_iteration == self.lastupdateatiter:
                    self.lastupdateatiter = self.tempering_training_iteration

                    if self.convergence_crit_type == 'check_loss_increase':
                        bgradcheck = self.check_conv() # original implementation
                    elif self.convergence_crit_type == 'check_loss_band':
                        bgradcheck = self.checkConvBoth()
                    elif self.convergence_crit_type == 'check_grad_norm':
                        bgradcheck, incline = self.check_conv_grad(gradient_norm, 200)
                    else:
                        raise ValueError('Please check if selected convergence criterion exists.')

                    if bgradcheck or self.enforceUpdateStep():
                        if self.eval_method_kl is 'Standard':
                            self.bresetOptimizer = True
                            self.converged_steps += 1
                            self.a = self.eval_a_function(kl_current, self.converged_steps)
                            self.writeHistory()
                        else:
                            self.a, perform_step, kl_increase = self.eval_a_function_md(kl_current, importance_sampler)
                            if perform_step:
                                self.bresetOptimizer = True
                                self.converged_steps += 1
                                importance_sampler.update_final_proposed_log_z_diff()
                                self.writeHistory(kl_increase, incline)

            if self.tempering_training_iteration % self.tempering_check_change_every == 0:
                if os.path.isfile(self.tempering_file):
                    self.a = np.loadtxt(self.tempering_file)
        else:
            self.a = 1.0

        return self.a

    def eval_a_function(self, kl_current, step):
        if self.assess_loss is None:
            raise ValueError('Loss assessment is not set. Please set this before.')

        current_factor = self.a
        proposed_factor = self.a
        relmax_kl_increase = 0.005 * self.relmax_kl_increase if self.a < 1.e-5 else self.relmax_kl_increase
        #relmax_kl_increase = self.relmax_kl_increase
        min_factor = 1.e-10
        f = 1.
        while f > min_factor:
            proposed_factor = current_factor + f * self.max_beta_increase
            kl_proposed = self.assess_loss(beta_prefactor=proposed_factor)
            if abs(kl_proposed - kl_current) / abs(kl_current) > relmax_kl_increase:
                f *= 0.6
            else:
                #if proposed_factor - current_factor > self.max_beta_increase:
                #    proposed_factor = self.max_beta_increase
                break
        if step <= 5:
            proposed_factor = current_factor * 3.
        return proposed_factor

    def eval_a_function_md(self, kl_current=None, importance_sampler=None):
        if self.assess_loss is None:
            raise ValueError('Loss assessment is not set. Please set this before.')

        do_step = True

        current_factor = self.a
        proposed_factor = self.a
        min_factor = 1.e-14
        f = 1.

        #max_beta_inc = 4. * self.a
        #max_beta_inc = max_beta_inc if max_beta_inc < self.max_beta_increase else self.max_beta_increase
        max_beta_inc = self.max_beta_increase
        while f > min_factor:
            proposed_factor = current_factor + f * max_beta_inc
            kl_increase = self.assess_loss(beta_pref_proposed=proposed_factor, beta_pref_current=current_factor)
            if kl_increase > self.relmax_kl_increase and kl_increase > 0.:
                f *= 0.6
            elif kl_increase <= 0.:
                proposed_factor = self.a
                do_step = False
                warnings.warn('KL increase is negative. KL increase is %.2f' % kl_increase, Warning)
                break
            else:
                #if proposed_factor - current_factor > self.max_beta_increase:
                #    proposed_factor = self.max_beta_increase
                self.count_too_small_f = 0
                break
        # check if the proposed f is smaller the the minimal accepted one. This should not happen - maybe it is not
        # converged.
        if f <= min_factor:
            self.count_too_small_f += 1
            if self.count_too_small_f < 5:
                proposed_factor = self.a
                do_step = False
                warnings.warn('Prefactor gets to small. Repeating with current temperature: %.2f' % self.a, Warning)
            else:
                self.count_too_small_f = 0

        return proposed_factor, do_step, kl_increase

class SchedulerBeta(SchedulerBetaParent):
    def __init__(self, a_init, a_end, checknlast, avginterval, expoffset, outputpath, angular, bLin=False, maxsteps=800, bresetopt=False, intwidth=None):
        super(SchedulerBeta, self).__init__(bresetopt=bresetopt, outputpath=outputpath, a_init=a_init,
                                                   a_end=a_end, angular=angular, intwidth=intwidth)
        self.bLin = bLin
        self.offset = expoffset
        self.nlast = checknlast
        self.obsinterval = avginterval

        self.maxsteps = maxsteps

    def eval_a_function(self, step):
        x = step
        maxstepexp = 9

        if self.bLin:
            if x >= self.maxsteps:
                a = 1.0
            else:
                a = self.a_init + (1. - self.a_init) / self.maxsteps * x
        else:
            if x >= self.offset - maxstepexp:
                if x >= (self.offset - maxstepexp + self.maxsteps):
                    a = 1.0
                else:
                    a0 = np.exp(-maxstepexp)
                    a = a0 + (self.a_end - a0)/self.maxsteps * (x - self.offset + maxstepexp)
            else:
                a = np.exp(x - self.offset)
        return a

    def checkMinIterations(self):
        conviterations = len(self.lastconvergediteration)
        if conviterations >= 2:
            iterationssincelastconv = self.lastconvergediteration[-1] - self.lastconvergediteration[-2]
            if iterationssincelastconv >= self.convCheckOptions['miniterationsperconv']:
                return True
            else:
                return False
        else:
            return True

    def updateLearningPrefactor(self, kl_current):
        if self.usesched:
            # check convergence
            if not (self.tempering_training_iteration % self.obsinterval):
                if not self.tempering_training_iteration == self.lastupdateatiter:
                    self.lastupdateatiter = self.tempering_training_iteration
                    if self.checkConvBoth() or self.enforceUpdateStep():
                        self.bresetOptimizer = True
                        self.converged_steps += 1
                        self.a = self.eval_a_function(self.converged_steps)
                        self.writeHistory()

            if self.tempering_training_iteration % self.tempering_check_change_every == 0:
                if os.path.isfile(self.tempering_file):
                    self.a = np.loadtxt(self.tempering_file)
        else:
            self.a = 1.0

class SchedulerBetaConvolution(SchedulerBetaParent):
    def __init__(self, a_init, a_end, checknlast, avginterval, expoffset, outputpath, angular, bLin=False, maxsteps=800, bresetopt=False, intwidth=None, max_sig_decrease=1.e-4):
        super(SchedulerBetaConvolution, self).__init__(bresetopt=bresetopt, outputpath=outputpath, a_init=a_init,
                                                   a_end=a_end, angular=angular, intwidth=intwidth)

        self.bLin = bLin
        self.offset = expoffset
        self.nlast = checknlast
        self.obsinterval = avginterval

        self.maxsteps = maxsteps
        self.usesched = False
        self.max_sig_decrease = max_sig_decrease

        self.step_history_file = os.path.join(outputpath, 'stepper_cov_history')

    def eval_a_function(self, step):
        # This should always return 1.0 since this is not responsible for tempering but for the convolution.
        return 1.0

    def evalConvSigma(self, step):
        x = step
        if self.bLin:
            if x >= self.maxsteps:
                sig_cov = self.a_end
            else:
                sig_cov = self.a_init - (self.a_init - self.a_end) / self.maxsteps * x
        else:
            raise NotImplementedError('Only linear decay is implemented for the convolution scheduler.')
        return sig_cov

    def evalConvSigmaKL(self, kl_current):
        if self.assess_loss is None:
            raise ValueError('Loss assessment is not set. Please set this before.')

        current_factor = self.a
        proposed_factor = self.a
        min_factor = 1.e-10
        f = 1.
        while f > min_factor:
            proposed_factor = current_factor - f * self.max_sig_decrease
            kl_proposed = self.assess_loss(beta_prefactor=proposed_factor)
            if abs(kl_proposed - kl_current) / abs(kl_current) > self.relmax_kl_increase:
                f *= 0.6
            else:
                #if proposed_factor - current_factor > self.max_beta_increase:
                #    proposed_factor = self.max_beta_increase
                break
        return proposed_factor

    def checkMinIterations(self):
        conviterations = len(self.lastconvergediteration)
        if conviterations >= 2:
            iterationssincelastconv = self.lastconvergediteration[-1] - self.lastconvergediteration[-2]
            if iterationssincelastconv >= self.convCheckOptions['miniterationsperconv']:
                return True
            else:
                return False
        else:
            return True

    def getconvergenceinterval(self, last_loss):
        interval = self.convCheckOptions['intvervalwidth'] * last_loss
        interval = 300 if interval > 300 else interval
        return interval

    def updateLearningPrefactor(self, kl_current):
        if self.usesched:
            raise ValueError('This scheduler is only for convolultion purposes.')

    def updateSigmaConv(self, kl_current):
        if self.a <= self.a_end:
            self.a = self.a_end
        else:
            # check convergence
            if not (self.tempering_training_iteration % self.obsinterval):
                if not self.tempering_training_iteration == self.lastupdateatiter:
                    self.lastupdateatiter = self.tempering_training_iteration
                    if self.check_conv() or self.enforceUpdateStep():
                        self.bresetOptimizer = True
                        self.converged_steps += 1
                        self.a = self.evalConvSigma(self.converged_steps)
                        #self.a = self.evalConvSigmaKL(kl_current)
                        self.writeHistory()

            if self.tempering_training_iteration % self.tempering_check_change_every == 0:
                if os.path.isfile(self.tempering_file):
                    self.a = np.loadtxt(self.tempering_file)

    def getLearningPrefactor(self):
        raise ValueError('This is a scheduler for sigma_cov for the convolution of the potential.')

    def getSigmaCov(self):
        return self.a

def plotStepSched():

    import matplotlib.pyplot as plt

    offset = 17
    maxstepexp = 3
    sOutputpath=''
    bs = SchedulerBeta(a_init=np.exp(-offset - 1), a_end=1., checknlast=4, avginterval=10, expoffset=offset, outputpath=sOutputpath, angular=False)

    k = np.arange(0, 81)
    ak = [bs.eval_a_function(kval) for kval in k]

    f, ax = plt.subplots(1)
    ax.semilogy(ak, lw='3')
    ax.axvline(x=offset - maxstepexp, ls='--', c='k', alpha=0.5, lw='4')
    ax.set_xlim(left=0)
    ax.grid(ls='dashed')
    ax.set_xlabel('k')
    ax.set_ylabel('a(k)')
    plt.savefig('step_off_17_maxexp_3_maxlinstep_50.pdf')
    plt.show()


import unittest

class TestSchedulerBeta(unittest.TestCase):

    def setUp(self):
        import argparse

        self.wd = os.getcwd()
        offset = 20
        maxschedsteps = 500
        self.schedBeta = SchedulerBeta(a_init=1.e-7, a_end=1., checknlast=4, avginterval=10,
                                       expoffset=offset, outputpath=self.wd,
                                       angular=False, bLin=True, maxsteps=maxschedsteps)

        self.schedBetaKL = SchedulerBetaKL()

    def test_miniterations_true(self):
        self.schedBeta.lastconvergediteration = []
        self.schedBeta.lastconvergediteration.append(100)
        self.schedBeta.lastconvergediteration.append(1000)
        bAssertTrue = self.schedBeta.checkMinIterations()
        self.assertTrue(bAssertTrue)

    def test_miniterations_false(self):
        self.schedBeta.lastconvergediteration = []
        self.schedBeta.lastconvergediteration.append(100)
        self.schedBeta.lastconvergediteration.append(102)
        bAssertFalse = self.schedBeta.checkMinIterations()
        self.assertFalse(bAssertFalse)