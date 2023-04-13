import torch
import numpy as np

import openmm as omm
import openmm.app as app
import openmm.unit as ommunit
from utils_peptide_torch import convertangularaugmenteddataset
from utils_peptide_torch import jacobiMatrixdxdxbar
from utils_peptide_torch import jacobiMatrixdxbardr
from utils_peptide_torch import convertToFullCartersianCoordinates

from SchedulerBeta import SchedulerBeta
from SchedulerBeta import SchedulerBetaKL
from SchedulerBeta import SchedulerBetaConvolution

# for scaling test purposes
from timeit import default_timer as timer

from openmmtools import testsystems as ommttestsystems

import os

from PropertyCal import PropertyCal

exit()
class MDImportanceSampler:
    def __init__(self, mdloss, mdrefmodel, q_dist, dim, dim_z, device=torch.device('cpu')):

        self.md_eval_potential = mdloss
        self.md_ref = mdrefmodel
        self.qdist = q_dist
        self.dim = dim
        self.device = device
        self.z_dim = dim_z
        self.z_current = None
        self.z_diff_current = None
        self.z_init_set = False

    def sample(self, breweight=False, beta_prefactor=1.0, n_z=2000, n_xpz=2, return_log_p_current=False):
        log_p_current = None
        nsamples = n_z * n_xpz
        data_z = torch.randn((n_z, self.z_dim), device=self.device)
        data_z_aug = data_z.repeat(n_xpz, 1)
        # mu and logvar of p(x|z)
        with torch.no_grad():
            data_x_rev_variational, q_recon_batch, pmu, plogvar = self.qdist.forward(data_z_aug)
        samples_q = data_x_rev_variational
        if return_log_p_current:
            log_w, log_p_current = self.log_w(samples_q, data_z_aug, beta_prefactor, return_log_p_current)
        else:
            log_w = self.log_w(samples_q, data_z_aug, beta_prefactor, return_log_p_current)

        if breweight:
            m = torch.distributions.Multinomial(nsamples, logits=log_w)
            msample = m.sample()
            index_q = torch.zeros(nsamples, device=self.device).long()
            count = 0
            for idx, amount in enumerate(msample):
                if amount > 0:
                    for j in range(int(amount.item())):
                        index_q[int(count)] = idx
                        count += 1
            return samples_q.index_select(0, index_q)
        else:
            if return_log_p_current:
                return samples_q, log_w, log_p_current
            else:
                return samples_q, log_w

    def mean(self, N=1000000):
        x, log_w = self.sample(N, breweight=False)
        m = (x * log_w.exp().unsqueeze(1)).sum(dim=0)
        return m

    def variance(self):
        raise NotImplementedError('Variance estimation is so far not implemented.')

    def log_w(self, x, z, beta_pref, return_log_p_current):
        # evaluate log q(x|z)
        #log_q = self.qdist.get_log_q_x_given_z(x, z)
        log_q = self.qdist.get_log_r_z_given_x(x, z)
        # evaluate -\beta U(x)
        log_p = self.md_eval_potential(x, self.md_ref, False, beta_pref, False)
        log_w = log_p - log_q
        log_w_norm = self.get_normalized_log_w(log_w)
        if return_log_p_current:
            return log_w_norm, log_p
        else:
            return log_w_norm

    def get_normalized_log_w(self, log_w):
        c = self.logsumexp(log_w)
        return log_w - c

    def logsumexp(self, log_x):
        """Perform the log-sum-exp of the weights."""
        max_exp = log_x.max()
        my_sum = torch.exp(log_x - max_exp).sum()
        return torch.log(my_sum) + max_exp

    def compute_log_diff_z_betaprime_m_zbeta(self, beta_pref_current, beta_prefactor_proposed):
        # Importance sampling of current distribution
        samples_q, log_w, m_a_i_beta_ux = self.sample(breweight=False, beta_prefactor=beta_pref_current, return_log_p_current=True)
        #N = torch.tensor(samples_q.shape[0], dtype=torch.float, device=self.device)
        # compute -\Delta \beta U(x)
        m_delta_beta_ux = (beta_prefactor_proposed - beta_pref_current) * m_a_i_beta_ux / beta_pref_current
        # preparing for log sum exp trick
        log_x = log_w + m_delta_beta_ux
        # MCMC estimator
        log_z_beta_proposed_m_z_beta_curr = self.logsumexp(log_x) #- torch.log(N)

        return log_z_beta_proposed_m_z_beta_curr

    def store_current_log_z_diff(self, log_z_diff):
        self.z_diff_current = log_z_diff

    def update_final_proposed_log_z_diff(self):
        if self.z_diff_current is None:
            raise ValueError('The current value of the log difference has not been set.')
        else:
            self.z_current += self.z_diff_current
            self.z_diff_current = None

    def get_current_log_z(self, beta_pref_current):
        # initialize if z_current hast not been initialized before
        if self.z_current is None:
            self.z_current = self.estimate_z_init(beta_pref_current)
        return self.z_current

    def estimate_z_init(self, beta_pref_current):

        # choose the amount of samples depending on the machine in use
        n_z = 100000 if self.device == torch.device('cuda') else 10000
        n_xpz = 10

        z = torch.randn((n_z, self.z_dim), device=self.device)
        z = z.repeat(n_xpz, 1)
        N = torch.tensor(z.shape[0], dtype=torch.float, device=self.device)
        with torch.no_grad():
            x, z_reconstructed, pmu, plogvar = self.qdist.forward(z)

        log_p = self.md_eval_potential(x, self.md_ref, False, beta_pref_current, False)
        log_r = self.qdist.get_log_r_z_given_x(samples_x=x, samples_z=z)
        log_qxgz = self.qdist.get_log_q_x_given_z(samples_x=x, samples_z=z)
        log_qz = self.qdist.get_log_q_z(samples_z=z)
        log_w = log_p + log_r - log_qxgz - log_qz
        log_Z_beta = self.logsumexp(log_w) - torch.log(N)

        log_W = self.get_normalized_log_w(log_w)
        ESS = 1. / log_W.exp().pow(2).sum().item()
        #w = log_w.exp()
        #ESS = w.sum().pow(2) / w.pow(2).sum()

        print('The ESS for the initial estimation of the normalization constant is {}'.format(ESS))
        #quit()
        return log_Z_beta






























































class MDSimulator:
    def __init__(self, pdbstructure, bGPU=False, sAngularRep='', sOutputpath='', nSimulators=1, stepschedopt=None,
                 breddescr=False, gradientpostproc='none', reference_configuration=None, gaussian_sig_sq=None,
                 convolve_pot_sig=None, convolve_pot_n=10, a_init_preset=None):

        # use standard openmm reference trajectory
        self.use_ommtools_reference = True
        self.pdb_structure_file = pdbstructure

        # reference trajectory xtc
        # TODO add the reference file name
        #reference_trj_xtc_file = os.path.join(pdbstructure.rsplit('/', 1)[0], 'mixed_data_10000_m_1527.xtc')
        reference_trj_xtc_file = os.path.join(pdbstructure.rsplit('/', 1)[0], 'output_sim_nvt_330.pdb')
        # object for calculating properties
        self.propertycal = PropertyCal(reference_trj_xtc_file, output_path=sOutputpath,
                                  pdb_ref_alpha_helix=pdbstructure, pdb_ref=pdbstructure)

        self.implicit_solvent = True
        self.beta_init_is_set = False
        convolve_pot_sig = 0. if convolve_pot_sig is None else convolve_pot_sig
        if convolve_pot_sig <= 0:
            self.conv_gaussian = False
        else:
            self.conv_gaussian = True
            self.conv_n = convolve_pot_n
            self.conv_sig = convolve_pot_sig

        self.reference_configuration = reference_configuration
        self.gaussian_sig_sq = gaussian_sig_sq
        if self.reference_configuration is None:
            self.use_reference_gaussian = False
        else:
            self.use_reference_gaussian = True
            if bGPU:
                self.reference_configuration = reference_configuration.to(torch.device('cuda'))

        if sAngularRep in 'ang_auggrouped':
            self.sCoordinateRep = 'AngularAugmented'
            self.jacobidxdxbar = jacobiMatrixdxdxbar(bGPU=bGPU)
        else:
            self.sCoordinateRep = 'Cartesian'
            # in this case we do not need the jacobian
            self.jacobidxdxbar = None

        if stepschedopt is None:
            self.bUseStepSched = False
        else:
            self.bUseStepSched = stepschedopt['usestepsched']
        self.outputPath = sOutputpath
        self.bContraintHBonds = False
        self.breddescr = breddescr
        self.bGPU = bGPU
        self.integratorlist = []
        self.simlist = []

        if bGPU:
            platform = omm.Platform.getPlatformByName('CUDA')
        else:
            platform = omm.Platform.getPlatformByName('CPU')

        # use either 'gradclamp' or 'gradnorm' or ''
        self.gradpostproc = gradientpostproc

        #self.bgradclamp = False
        self.gradClamp = 1.e15

        #self.bgradrenorm = True

        # set a_init
        a_init = a_init_preset if a_init_preset is not None else 1e-10


        #a_init=1.e-24
        if sAngularRep in 'ang_auggrouped':
            # the angular representation is more sensitive
            offset = 20.
        else:
            offset = 17
            #offset = 10.

        #self.schedBeta = SchedulerBeta(a_init=np.exp(-offset - 1), a_end=1., checknlast=4, avginterval=10,
        #                               expoffset=offset, outputpath=sOutputpath,
        #                               angular=(sAngularRep in 'ang_auggrouped'),
        #                               bresetopt=stepschedopt['stepschedresetopt'])
        if stepschedopt.get('sched_method_check_convergence') is None:
            sched_method_check_convergence = 'check_loss_increase'
        else:
            sched_method_check_convergence = stepschedopt.get('sched_method_check_convergence')

        if stepschedopt is not None:
            if stepschedopt['stepschedtype'] in 'lin':
                self.schedBeta = SchedulerBeta(a_init=a_init, a_end=1., checknlast=4, avginterval=10,
                                               expoffset=offset, outputpath=sOutputpath,
                                               angular=(sAngularRep in 'ang_auggrouped'),
                                               bLin=True, maxsteps=stepschedopt['imaxstepssched'],
                                               bresetopt=stepschedopt['stepschedresetopt'],
                                               intwidth = stepschedopt['stepschedconvcrit'])
                self.schedBeta.setSched(self.bUseStepSched, a_init_preset)
            elif stepschedopt['stepschedtype'] in 'kl':
                self.schedBeta = SchedulerBetaKL(a_init=a_init,
                                                 #a_init=1.e-18,
                                                 outputpath=sOutputpath,
                                                 bresetopt=stepschedopt['stepschedresetopt'], max_beta_increase=1.e-4,
                                                 eval_method_kl='Standard',
                                                 #eval_method_kl='MD',
                                                 convergence_crit_type=sched_method_check_convergence,
                                                 max_kl_inc=1.0)#2.e-7)#max_kl_inc=0.000025)
                self.schedBeta.setSched(self.bUseStepSched, a_init_preset)
            else:
                raise NotImplementedError('Scheduler option not implemented.')

        if self.conv_gaussian:
            self.schedBetaCov = SchedulerBetaConvolution(a_init=0.2, a_end=1.e-7, checknlast=4, avginterval=10,
                                           expoffset=offset, outputpath=sOutputpath,
                                           angular=(sAngularRep in 'ang_auggrouped'),
                                           bLin=True, maxsteps=500,
                                           bresetopt=False,
                                           intwidth=stepschedopt['stepschedconvcrit'])
            self.schedBetaCov.setSched(False)


        self.temp = 330.
        self.PDBstructureFile = pdbstructure
        self.forcefieldoptions = '\'amber96.xml\', \'amber96_obc.xml\''
        self.pdb = app.PDBFile(self.PDBstructureFile)
        #self.forcefield = app.ForceField(eval(self.forcefieldoptions)[0])
        if self.implicit_solvent:
            self.forcefield = app.ForceField('amber96.xml', 'amber96_obc.xml')
        else:
            self.forcefield = app.ForceField('amber96.xml')

        # this section is for using the openmmtools reference simulation system
        if self.use_ommtools_reference:
            if self.bContraintHBonds:
                self.mdtestobj = ommttestsystems.AlanineDipeptideImplicit()
            else:
                self.mdtestobj = ommttestsystems.AlanineDipeptideImplicit(constraints=None)
            # define the mdsystem for later use
            self.mdsystem = self.mdtestobj.system

        # this section for creating the system manually
        else:
            if self.bContraintHBonds:
                self.mdsystem = self.forcefield.createSystem(self.pdb.topology, nonbondedMethod=omm.app.NoCutoff,
                                                             constraints=omm.app.HBonds, soluteDielectric=1.0,
                                                             solventDielectric=80.0)
            else:
                if self.implicit_solvent:
                    self.mdsystem = self.forcefield.createSystem(self.pdb.topology, nonbondedMethod=omm.app.NoCutoff,
                                                             soluteDielectric=1.0,
                                                             solventDielectric=80.0)
                else:
                    self.mdsystem = self.forcefield.createSystem(self.pdb.topology, nonbondedMethod=omm.app.NoCutoff)
                #self.mdsystem = self.forcefield.createSystem(self.pdb.topology, nonbondedMethod=omm.app.NoCutoff #CutoffNonPeriodic,#NoCutoff,
                #                                             soluteDielectric=1.0,
                #                                             solventDielectric=80.0)

        self.createSimObjectList(ommpdb=self.pdb.topology, ommmdsystem=self.mdsystem, nsimulators=1, platform=platform)

        self.mdintegrator = omm.LangevinIntegrator(self.temp*ommunit.kelvin, 1/ommunit.picosecond, 0.002*ommunit.picoseconds)

        if bGPU:
            self.simulation = app.Simulation(self.pdb.topology, self.mdsystem, self.mdintegrator, platform)
        else:
            self.simulation = app.Simulation(self.pdb.topology, self.mdsystem, self.mdintegrator)

    def checksetoptions(self):
        # TODO implement chekcs where required.
        pass

    def createSimObjectList(self, ommpdb, ommmdsystem, nsimulators, platform):

        self.integratorlist = [omm.LangevinIntegrator(self.temp * ommunit.kelvin, 1 / ommunit.picosecond,
                                                 0.002 * ommunit.picoseconds) for m in range(nsimulators)]

        self.simlist = [app.Simulation(ommpdb, ommmdsystem, self.integratorlist[m], platform) for m in
                        range(nsimulators)]

    def doStep(self):
        if hasattr(self, 'schedBeta'):
            self.schedBeta.doStep()
        if hasattr(self, 'schedBetaCov'):
            self.schedBetaCov.doStep()

    def getModelType(self):
        return 'MD'

    def getTemperature(self):
        return self.temp

    def getBeta(self, force_prefactor=None):
        if force_prefactor is None:
            if hasattr(self, 'schedBeta'):
                return 1./(self.temp*ommunit.kelvin * self.getBoltzmannConstant()) * self.schedBeta.getLearningPrefactor()
            else:
                return 1./(self.temp*ommunit.kelvin * self.getBoltzmannConstant()) * 1.
        else:
            return 1. / (
                        self.temp * ommunit.kelvin * self.getBoltzmannConstant()) * force_prefactor

    def getBoltzmannConstant(self):
        return ommunit.constants.BOLTZMANN_CONSTANT_kB*ommunit.constants.AVOGADRO_CONSTANT_NA

    def reducedDescription(self, inputTorch):
        with torch.no_grad():
            if self.breddescr:
                x = convertToFullCartersianCoordinates(data=inputTorch)
            else:
                x = inputTorch
        return x

    def reducedDescriptionForceConversion(self, f, dofsnp=np.array([18, 19, 20, 24, 25, 43], dtype=int), x_dim_red=60, x_dim_original=66):
        if self.breddescr:
            dofs = torch.from_numpy(dofsnp).long()
            mask = torch.ones(f.shape[0], f.shape[1]).byte()
            for i in dofs:
                mask[:, i] = 0
            f_red = f[mask].unsqueeze_(1).view(-1, x_dim_red)
        else:
            f_red = f

        return f_red

    def setposzero(self, np_pos, dofsnp=np.array([18, 19, 20, 24, 25, 43])):
        for dof in dofsnp:
            np_pos[int(dof / 3), dof % 3] = 0.

    def convertToCartesian(self, inputTorch):

        if not self.sCoordinateRep in 'Cartesian':
            lengDim0 = inputTorch.shape[0]
            lengDim1 = inputTorch.shape[1]
            nAtoms = int(lengDim1 / 5 + 1)

            # # convert to coordinates x ndatapoints
            # if self.bGPU:
            #     npinput = inputTorch.data.cpu().numpy()
            # else:
            #     npinput = inputTorch.data.numpy()

            # in case we use internally different coordinates than the cartesian representation we need to convert
            # them to cartesian coordinates for openMM
            # TODO CALCULATE JACOBIAN FOR dr/dx and account for it in backward
            torchCoord = convertangularaugmenteddataset(inputTorch.transpose(dim0=1, dim1=0), bgrouped=True)

            #torchCoordTemp = torch.from_numpy(npCartesian.T)

            # if self.bGPU:
            #     torchCoord = torchCoordTemp.to('cuda')
            # else:
            #     torchCoord = torchCoordTemp
            #atomisticView = torchCoord.view([lengDim0, nAtoms, 3])
            atomisticView = torchCoord.permute(2, 0, 1)

        else:
            x = self.reducedDescription(inputTorch)
            lengDim0 = x.shape[0]
            lengDim1 = x.shape[1]
            nAtoms = int(lengDim1 / 3)
            atomisticView = x.view([lengDim0, nAtoms, 3])

        return atomisticView, nAtoms, lengDim0

    def convolvegradient(self, np_frame, beta):
        if hasattr(self, 'schedBetaCov'):
            sig = self.schedBetaCov.getSigmaCov()
        else:
            sig = self.conv_sig
        dim0 = np_frame.shape[0]
        dim1 = np_frame.shape[1]
        force = np.zeros([dim0, dim1])
        potential = 0.
        for i in range(self.conv_n):
            np_frame_sample = np.random.randn(dim0, dim1) * sig + np_frame
            self.setposzero(np_frame_sample)
            # set here dofs to zero
            self.simlist[0].context.setPositions(np_frame_sample)
            reporter = self.simlist[0].context.getState(getForces=True, getEnergy=True)
            frep = reporter.getForces(asNumpy=True)
            if i == 0:
                force = frep
                potential = reporter.getPotentialEnergy()
            else:
                force += frep
                potential += reporter.getPotentialEnergy()
        ftemp = -(beta * force)
        ftemp /= (ftemp.unit * self.conv_n)
        potential = -beta * potential / self.conv_n

        return ftemp, potential

    def check_nan(self, **kwargs):
        for key, value in kwargs.items():
            if torch.isnan(value).any():
                print('Nan in variable {}'.format(key))
                raise ValueError('Nan occured.')
        return True

    def postprocessgradient(self, gradient, potential=None, max_norm=10, itera=0):
        atoms = gradient.shape[1]
        dofperatom = gradient.shape[2]
        dofs = atoms * dofperatom
        gtemp = gradient.view(-1, dofs)
        with torch.no_grad():
            if False: #'gradnormsingle' in self.gradpostproc:
                gradnormvals = gtemp.norm(dim=1, keepdim=True)
                gtemp = gtemp/gradnormvals
                return gtemp.unsqueeze(2).view(-1, atoms, dofperatom)
            elif 'gradnormsingle' in self.gradpostproc:
                gradnormvals = gtemp.norm(dim=1, keepdim=True)
                mean_grad_norm = gradnormvals.mean()
                # original implementation with 1.5
                # max_norm = mean_grad_norm * 20.
                max_norm = mean_grad_norm * 1.5

                if itera % 500 == 0:
                    filename = os.path.join(self.outputPath, 'grad_norm_{}.txt'.format(itera))
                    np.savetxt(filename, gradnormvals.data.cpu().numpy())

                #if max_norm > 1.e5:
                #    max_norm = 1.e5
                #individually check if norm larger than threshold
                gnormmult = gradnormvals > max_norm
                clip_coef = gnormmult.type(torch.float) * gradnormvals
                clip_coef[clip_coef == 0] = max_norm
                clip_coef = max_norm / clip_coef
                if (clip_coef < 0.999).any():
                    gtemp.mul_(clip_coef)
                    if potential is not None:
                        potential.mul_(clip_coef.squeeze())

                self.check_nan(gradient=gtemp, gradnorm=gradnormvals, clip_co=clip_coef)
                return gtemp.unsqueeze(2).view(-1, atoms, dofperatom)
            elif 'gradnormmax' in self.gradpostproc:
                gradnormvals = gtemp.norm(dim=1, keepdim=True)
                total_norm = (gradnormvals ** 2.).sum()
                total_norm = total_norm ** (1. / 2) # L2 Norm
                clip_coef = max_norm / (total_norm + 1e-6)
                if clip_coef < 1:
                    gtemp.mul_(clip_coef)
                return gtemp.unsqueeze(2).view(-1, atoms, dofperatom)

            elif 'gradnorm' in self.gradpostproc:
                gradnormvals = gtemp.norm(dim=1, keepdim=True)
                gradnormmax = gradnormvals.max()
                gtemp = gtemp / gradnormmax
                return gtemp.unsqueeze(2).view(-1, atoms, dofperatom)
            elif 'gradclamp' in self.gradpostproc:
                self.clampGrad(gtemp, 5.)
                gtemp.clamp_(min=-self.gradClamp, max=self.gradClamp)
                return gtemp.unsqueeze(2).view(-1, atoms, dofperatom)
            elif 'monitor' in self.gradpostproc:
                if itera % 500 == 0:
                    gradnormvals = gtemp.norm(dim=1, keepdim=True)
                    filename = os.path.join(self.outputPath, 'grad_norm_{}.txt'.format(itera))
                    np.savetxt(filename, gradnormvals.data.cpu().numpy())
                return gradient
            elif 'none' in self.gradpostproc:
                return gradient
            else:
                raise ValueError('Invalid gradient postprocessing method used.')

    @staticmethod
    def clampGrad(input, factor=3.):
        with torch.no_grad():
            absmean = input.abs().mean()
            clampVal = absmean*factor
            input.clamp_(min=-clampVal, max=clampVal)

    def setBetaInit(self, beta):
        if not self.beta_init_is_set:
            self.beta_init = beta
            self.beta_init_is_set = True

    def getBetaInit(self):
        if self.beta_init_is_set:
            return self.beta_init
        else:
            raise ValueError('Beta was not initially set')






























































class MDLoss(torch.autograd.Function):
    '''
    This class is a loss function log p(x) of Boltzmann distribution.
    '''
    @staticmethod
    def forward(ctx, input, mdsimulator, reduce=False, beta_prefactor=None, calc_forces=True):
        # The usual forces in a proper peptide are in the range +/- 5x10e3 therefore clamp forces exceeding 10e5
        bound_forces = 0. # 1.e20#0.#1.e8
        carbox_potential = False
        carbox_factor = 1.e2
        max_box_size = 2.5
        bTimer = False

        if bTimer:
            start = timer()

        if MDLoss.checkNanInput(input):
            print(input)

            #input_prev, u_prev, dudx_prev = ctx.saved_tensors
            #np.savetxt('force_nan_inf.txt', dudx_prev.data.numpy())

        # represent the input vector x in a multiple of 3 for cartesian coordinates
        # each row should relate to the coordinates of an atom.
        #
        atomisticView, nAtoms, lengDim0 = mdsimulator.convertToCartesian(inputTorch=input)

        # get beta
        if beta_prefactor is None:
            beta = mdsimulator.getBeta()
        else:
            beta = mdsimulator.getBeta(force_prefactor=beta_prefactor)

        if calc_forces:
           dudxnp = np.zeros([lengDim0, nAtoms, 3])
        unp = np.zeros(lengDim0)
        if mdsimulator.bGPU:
            atomisticViewNp = atomisticView.data.cpu().numpy()
        else:
            atomisticViewNp = atomisticView.data.numpy()

        # # convert from A to NM since OpenMM expects the input in NM
        # atomisticViewNp = atomisticViewNp * 0.1

        # split the atomisticViewNp into a list of array - each entry for one sample and iterate through them
        for idx, frame in enumerate(np.split(atomisticViewNp, lengDim0, axis=0)):
            npframe = frame.squeeze()

            if mdsimulator.conv_gaussian:
                ftemp, utemp = mdsimulator.convolvegradient(npframe, beta)
            else:
                mdsimulator.simlist[0].context.setPositions(npframe)

                if calc_forces:
                    reporter = mdsimulator.simlist[0].context.getState(getForces=True, getEnergy=True)
                    frep = reporter.getForces(asNumpy=True)
                    #print('Max val ', np.abs(frep / frep.unit).max())
                    #print('Avg val ', np.abs(frep / frep.unit).mean())
                    #print(frep)
                    MDLoss.checkNanForces(input=frep)
                    frep = MDLoss.limit_single_force_components(frep, bound_forces) if bound_forces > 1. else frep
                    #if (frep/frep.unit > 1.e10).any():
                        #print('Large forces occur.')
                    ftemp = -(beta * frep)
                    ftemp /= ftemp.unit
                else:
                    reporter = mdsimulator.simlist[0].context.getState(getEnergy=True)

                utemp = reporter.getPotentialEnergy()
                utemp = -beta*utemp
                #print 'Potential energy {0}'.format(unp[idx])

            # store the force and potential
            if calc_forces:
                dudxnp[idx, :, :] = np.copy(ftemp)
            unp[idx] = utemp

        u = torch.from_numpy(unp).float()
        if calc_forces:
            dudx = torch.from_numpy(dudxnp).float()
        if mdsimulator.bGPU:
            u = u.to('cuda')
            if calc_forces:
                dudx = dudx.to('cuda')

        # box the target potential
        if carbox_potential:
            beta_curr = beta / beta.unit
            mdsimulator.setBetaInit(beta_curr)
            beta_init = mdsimulator.getBetaInit()
            beta_boxpot_pref = beta_curr / beta_init
            # ToDo move this as option in the beginning of the function.
            beta_boxpot_pref = 1.

            u_carbox_cont, f_carbox_cont = MDLoss.add_boxcar_potential(input, max_box_size, beta_boxpot_pref,
                                                                       carbox_factor)
            u += u_carbox_cont

        if reduce or input.dim() == 1:
            output = u.sum()
        else:
            output = u

        if calc_forces:
            if hasattr(mdsimulator, 'schedBeta'):
                dudx = mdsimulator.postprocessgradient(gradient=dudx,
                                                       itera=mdsimulator.schedBeta.tempering_training_iteration)

            # push the calculation of grad here
            if not mdsimulator.sCoordinateRep in 'Cartesian':
                with torch.no_grad():
                    j1 = mdsimulator.jacobidxdxbar
                    j2 = jacobiMatrixdxbardr(input, bGPU=mdsimulator.bGPU)
                    #jcomb = torch.matmul(j1, j2)
                    gtemp = torch.matmul(-dudx.view([-1, nAtoms * 3]), j1)
                    gtemp = gtemp.unsqueeze(dim=1)

                    gforward = (torch.matmul(gtemp, j2)).squeeze()

                    MDLoss.clampGrad(gforward, 15.)
            else:
                with torch.no_grad():
                    gforward = -dudx.view([-1, nAtoms * 3])
                    gforward = mdsimulator.reducedDescriptionForceConversion(f=gforward)

            if carbox_potential:
                gforward += f_carbox_cont
                #idx_out_box_f = (input > max_box_size)
                ## this is the negative force, so F=-dU/dx therefore signs are flipped here
                #gforward[idx_out_box_f] = -carbox_factor * beta_boxpot_pref
                #idx_out_box_f = (input < -max_box_size)
                #gforward[idx_out_box_f] = carbox_factor * beta_boxpot_pref

            # add harmonic potential
            if mdsimulator.gaussian_sig_sq is not None:
                with torch.no_grad():
                    # standard gaussian
                    if not mdsimulator.use_reference_gaussian:
                        gforward -= 1./mdsimulator.gaussian_sig_sq * input
                    # gaussian around reference configuration
                    else:
                        gforward -= 1. / mdsimulator.gaussian_sig_sq * (input - mdsimulator.reference_configuration)


            ctx.save_for_backward(u, gforward)
            ctx.reduce = reduce
            ctx.bDebug = False
            ctx.nAtoms = nAtoms

            if bTimer:
                vec_time = timer() - start
                print(vec_time)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        u, gforward = ctx.saved_tensors

        grad = gforward

        grad_input = grad_output*grad

        if ctx.bDebug:
            print('Grad_output')
            print(grad_output)
            print('Grad_input')
            print(grad_input)

        return grad_input, None, None, None, None

    @staticmethod
    def limit_single_force_components(f, limit):
        f_unit = f.unit
        f = f/f_unit

        exceed_lim = abs(f) > limit
        if exceed_lim.any():
            not_exceed_lim = abs(exceed_lim - 1) * 1.
            sign = np.sign(f)
            exceed_sign = exceed_lim * sign

            f *= not_exceed_lim
            f += exceed_sign * limit

        f = f*f_unit
        return f

    @staticmethod
    def checkNanInput(input):
        nans = torch.isnan(input)
        nanentries = nans.nonzero()
        if nanentries.nelement() > 0:
            #np.savetxt('nanout.txt', input.data.numpy())
            return True
        else:
            return False

    @staticmethod
    def checkNanForces(input):
        '''
        check if any force is NAN or INF
        :param input:
        :return:
        '''
        maxval = 10.e19
        nanval = 1.e-5
        # NAN
        boolnanpos = np.isnan(input)
        if boolnanpos.any():
            print('NaN in returned forces from openMM.')
            inputunrolled = input.reshape(input.size)
            index = np.where(np.isnan(inputunrolled))
            inputunrolled[index] = nanval * input.unit
        #np.savetxt('force_nan_inf.txt', input)
        # INF
        boolinfpos = np.isinf(input)
        if boolinfpos.any():
            inputunrolled = input.reshape(input.size)
            index = np.where(np.isinf(inputunrolled))
            inputunrolled[index] = np.sign(inputunrolled[index]) * maxval * input.unit
            print('INF in returned forces from openMM.')
        #np.savetxt('force_nan_inf.txt', input)
        #input[bool]

    @staticmethod
    def add_boxcar_potential(input, max_box_size, beta_boxpot_pref, carbox_factor):
        # count_idx_out_box = (input > max_box_size).sum(dim=1)
        # u_add_multiplyer = count_idx_out_box.type_as(u)
        # u = u - carbox_factor * u_add_multiplyer * beta_boxpot_pref
        # count_idx_out_box = (input < -max_box_size).sum(dim=1)
        # u_add_multiplyer = count_idx_out_box.type_as(u)
        # u = u - carbox_factor * u_add_multiplyer * beta_boxpot_pref

        f_contribution = torch.zeros_like(input)

        # Heaviside function
        H_x = (input > max_box_size).type_as(input)
        #if H_x.type(torch.uint8).any():
        #    print('wait')
        # H_x times input
        H_x_mul_x = H_x * (input - max_box_size)
        # -beta u(x) contribution
        m_beta_u_contribution = - carbox_factor * beta_boxpot_pref * H_x_mul_x.sum()
        f_contribution = -carbox_factor * beta_boxpot_pref * H_x


        # Heaviside function
        H_x = (input < -max_box_size).type_as(input)
        #if H_x.type(torch.uint8).any():
        #    print('wait')
        # H_x times input
        H_x_mul_x = H_x * -(input + max_box_size)
        # -beta u(x) contribution
        m_beta_u_contribution -= carbox_factor * beta_boxpot_pref * H_x_mul_x.sum()
        f_contribution += carbox_factor * beta_boxpot_pref * H_x

        # idx_out_box = (input > max_box_size).any(1)
        # u[idx_out_box] = -1000.
        # idx_out_box = (input < -max_box_size).any(1)
        # u[idx_out_box] = -1000.

        return m_beta_u_contribution, f_contribution
















import unittest

class TestMDLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(138323)
        self.mdsim = MDSimulator('/home/schoeberl/Dropbox/PhD/projects/2018_07_06_openmm/ala2/ala2_adopted.pdb')
        self.mddata = np.loadtxt('/home/schoeberl/Dropbox/PhD/projects/2018_01_24_traildata_yinhao_nd/data_peptide/dataset_10.txt').T
        nframes = self.mddata.shape[0]
        np.expand_dims(self.mddata, axis=2)
        self.mddata_frame_rep = np.reshape(self.mddata, (nframes, 22, 3))

        dat = torch.rand(2, 66)
        self.x = dat
        self.x.requires_grad = True

        self.f = MDLoss.apply
        self.y = self.f(self.x, self.mdsim, True)
        self.z = self.y.pow(2.)

    def test_backward(self):
        # calculate gradient
        self.z.backward(retain_graph=True)
        # print gradient calculation
        print(torch.autograd.grad(self.z, self.y, retain_graph=True))
        print(torch.autograd.grad(self.y, self.x, retain_graph=True))
        print(torch.autograd.grad(self.z, self.x, retain_graph=True))
        # check if gradient has been calculated
        self.assertFalse( self.x.grad is None )

    def test_numerics(self):
        test = torch.autograd.gradcheck(self.f, (self.x, self.mdsim, True), eps=1e-3, atol=1e-3)
        self.assertTrue(True)

    def test_openmmforceevaluation(self):
        simulator = self.mdsim.simlist[0]

        for i, coord in enumerate(self.mddata_frame_rep):
            upot = 0.
            for j in range(1000):
                simulator.context.setPositions(coord)
                reporter = simulator.context.getState(getForces=True, getEnergy=True)
                upotmo = upot
                upot = reporter.getPotentialEnergy()
                print(upot)
                if j > 0:
                    if not upot == upotmo:
                        print('Potential energy is inconsistent.')
                        break

    #def test_gradcheck(self):
    #    test = torch.autograd.gradcheck(f, (x, mdsim, True), eps=1e-3, atol=1e-3)
