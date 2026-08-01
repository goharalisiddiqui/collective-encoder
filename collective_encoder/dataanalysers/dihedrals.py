import numpy as np

from .labels import LabelsAnalyser
from .datapoints import DatapointsAnalyser
from .graphs import GraphDatapointsAnalyser


from collective_encoder.testplotters.labels_selector import cos_sin_to_angle

class LabelsDihedralAnalyser(LabelsAnalyser):
    """
    Data analyser for extracting and plotting dihedral angles.
    """

    _IDENTIFIER = "LABELS_DIHEDRAL"

    def _plot_axes_modifier(self, ax, yonly=True):
        ax.set_ylim([-np.pi, np.pi])
        ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
        if not yonly:
            ax.set_xlim([-np.pi, np.pi])
            ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
            ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    
class DatapointsDihedralAnalyser(DatapointsAnalyser):
    """
    Data analyser for extracting and plotting dihedral angles.
    """

    _IDENTIFIER = "DATAPOINTS_DIHEDRAL"
    
    def _plot_axes_modifier(self, ax, yonly=True):
        return LabelsDihedralAnalyser._plot_axes_modifier(self, ax, yonly=yonly)

class GraphDatapointsDihedralAnalyser(GraphDatapointsAnalyser):
    """
    Data analyser for extracting and plotting dihedral angles.
    """

    _IDENTIFIER = "GRAPH_DATAPOINTS_DIHEDRAL"
    
    def _plot_axes_modifier(self, ax, yonly=True):
        return LabelsDihedralAnalyser._plot_axes_modifier(self, ax, yonly=yonly)
    
    def _extract_labels(self, data):
        labels = super()._extract_labels(data)
        dihedrals = cos_sin_to_angle(labels)
        return dihedrals

