
import numpy as np
import mdtraj as md
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import os

def calculate_phi_psi(trj, output_plot_file, bins=15):
    phi = md.compute_phi(trj)
    psi = md.compute_psi(trj)
    phipsi= np.hstack((phi[1],psi[1])) * 180 / np.pi

    # create the plot
    f, ax = plt.subplots()
    counts, xedges, yedges, im = ax.hist2d(phipsi[:,0], phipsi[:,1], bins=(bins, bins), normed=True)
    f.colorbar(im)
    ax.set_xlabel(r'$\phi$ [Degree]')
    ax.set_ylabel(r'$\psi$ [Degree]')
    f.savefig(output_plot_file, tight_layout=True)
    plt.close(f)

def calculate_bonded_statistics(traj_ref, traj_pred, output_plot_file):

    table, bonds_old = traj_ref.topology.to_dataframe()
    bonds = bonds_old[:, 0:2]
    topol = traj_ref.topology

    fig_size = 15
    n_rows = 5
    n_cols = 5

    bond_dist_ref = md.compute_distances(traj_ref, bonds) * 10
    bond_dist_pred = md.compute_distances(traj_pred, bonds) * 10

    n_bonds = bond_dist_ref.shape[1]

    f, ax = plt.subplots(n_rows, n_cols,
                         sharex=True, sharey=True, figsize=(fig_size, fig_size))
    f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.35, hspace=0.5)
    for i in range(n_rows):
        for j in range(n_cols):
            count = int(i * n_cols + j)
            if count >= n_bonds:
                ax[i, j].remove()
                # break
            else:
                ref = ax[i, j].hist(bond_dist_ref[:, count], label='Reference', density=True)
                pred = ax[i, j].hist(bond_dist_pred[:, count], label='Predicted', alpha=0.7, density=True)
                name_0 = topol.atom(int(bonds[count, 0])).name
                name_1 = topol.atom(int(bonds[count, 1])).name
                ax[i, j].set_title(name_0 + '-' + name_1)
                ax[i, j].set_xlim([0, 2])
                start, end = ax[i, j].get_xlim()
                stepsize = 0.5
                ax[i, j].xaxis.set_ticks(np.arange(start, end + stepsize, stepsize))
                # ax[i,j].grid(ls='dashed')
                # ax[i,j].set_axisbelow(True)
                if i == (n_rows - 1):
                    ax[i, j].set_xlabel(r'Bondlength [$\AA$]')
                # ax[i,j].legend()
    f.legend([ref, pred], labels=['Reference', 'Predicted'],
             loc='upper center', bbox_to_anchor=(0.5, 0.2), ncol=2)
    plt.savefig(output_plot_file)  # , tight_layout=True)
    plt.close(f)


class PropertyCal:
    def __init__(self, trj_ref, output_path=os.getcwd(), pdb_ref_alpha_helix=None, pdb_ref=None):
        self.file_trj_ref = trj_ref
        self.pdb_ref = pdb_ref
        self.output_path = output_path

        # reference trajectory object
        self.trj_ref = None
        self.load_reference()

        self.alpha_helix_reference_file = pdb_ref_alpha_helix
        self.trj_rmsd_ref = None
        self.rmsd_ref = None
        # load the rmsd reference and estimate the rmsd for test trajectory
        if self.alpha_helix_reference_file is not None:
            try:
                self.trj_rmsd_ref = md.load(self.alpha_helix_reference_file)
            except:
                raise FileExistsError('Error while loading alpha helix reference.')
            self.rmsd_ref = md.rmsd(self.trj_ref, self.trj_rmsd_ref)

        # calculate the reference radius of gyration
        self.rg_ref = md.compute_rg(self.trj_ref)

    def load_reference(self):
        # load the reference trajectory - with our without pdb file
        try:
            if (self.pdb_ref is None) or (self.file_trj_ref.rsplit('.', 1)[1] == 'pdb'):
                self.trj_ref = md.load(self.file_trj_ref)
            else:
                self.trj_ref = md.load(self.file_trj_ref, pdb=self.pdb_ref)
        except:
            raise IOError('Not able to load reference trajectory.')

    def write_trajectory(self, input_xyz_nparray, output_file):
        '''
        Write a numpy array to a compressed xtc file for further treatment.
        :param input_xyz_nparray: Numpy array with the dimension (n_atoms*3, N) with N the amount of samples
        :param output_file: String specifying the output file including the path.
        :return: True if successful
        '''
        dofs = input_xyz_nparray.shape[0]
        if dofs != 66:
            raise ValueError('This input does not correspond to an ALA2 peptide.')
        leng_trj = input_xyz_nparray.shape[1]
        with md.formats.XTCTrajectoryFile(output_file, 'w') as f:
            for pos in np.split(input_xyz_nparray, leng_trj, axis=1):
                f.write(pos.reshape(22, 3))
        f.close()

        return True

    def estimate_properties(self, prediction_xtc, output_prefix='', pdb_file=None):
        file_format = 'png'
        file_extension = prediction_xtc.rsplit('.', 1)[-1]
        if not (file_extension == 'xtc'):
            raise IOError('Input file should be an xtc trajectory.')

        if pdb_file is None:
            trj_pred = md.load(prediction_xtc, top=self.pdb_ref)
        else:
            raise NotImplementedError('Please check this implementation.')

        # bonded statistics
        output_bonded = os.path.join(self.output_path, output_prefix) + '_plot_bonded.{}'.format(file_format)
        calculate_bonded_statistics(self.trj_ref, trj_pred, output_bonded)

        # phi psi plot
        output_phi_psi = os.path.join(self.output_path, output_prefix) + '_plot_phi_psi.{}'.format(file_format)
        calculate_phi_psi(trj_pred, output_phi_psi)

        # rmsd
        output_rmsd = os.path.join(self.output_path, output_prefix) + '_plot_rmsd.{}'.format(file_format)
        if self.alpha_helix_reference_file is not None:
            rmsd = md.rmsd(trj_pred, self.trj_rmsd_ref)
            f, ax = plt.subplots()
            ax.hist(rmsd, density=True, label='Prediction', )
            if self.rmsd_ref is not None:
                ax.hist(self.rmsd_ref, density=True, label='Reference', alpha=0.3)
            ax.legend()
            f.savefig(output_rmsd, tight_layout=True)
            plt.close(f)

        # radius of gyration
        output_rg = os.path.join(self.output_path, output_prefix) + '_plot_rg.{}'.format(file_format)
        rg = md.compute_rg(trj_pred)
        f, ax = plt.subplots()
        ax.hist(rg, density=True, label='Prediction', )
        if self.rg_ref is not None:
            ax.hist(self.rg_ref, density=True, label='Reference', alpha=0.3)
        ax.legend()
        f.savefig(output_rg, tight_layout=True)
        plt.close(f)