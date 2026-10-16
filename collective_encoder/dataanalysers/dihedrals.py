import numpy as np

from .labels import LabelsAnalyser
from .datapoints import DatapointsAnalyser
from .graphs import GraphDatapointsAnalyser

from collective_encoder.testplotters.utils import cos_sin_to_angle

class LabelsDihedralAnalyser(LabelsAnalyser):
    """
    Data analyser for plotting dihedral angles fetched from labels.

    This analyser modifies the axes to reflect circular angular values
    between -pi and pi when creating plots based on labels.
    """

    _IDENTIFIER = "LABELS_DIHEDRAL"

    def _plot_axes_modifier(self, ax, yonly=True):
        """
        Adjust plot axes to properly display dihedral angle values (from -pi to pi).

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to modify.
        yonly : bool, optional
            Whether to apply the modifications only to the y-axis, by default True.
        """
        ax.set_ylim([-np.pi, np.pi])
        ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
        if not yonly:
            ax.set_xlim([-np.pi, np.pi])
            ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
            ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    
class DatapointsDihedralAnalyser(DatapointsAnalyser):
    """
    Data analyser for extracting and plotting dihedral angles from datapoints.

    Inherits from DatapointsAnalyser but modifies the plot axes specifically
    for circular dihedral measurements.
    """

    _IDENTIFIER = "DATAPOINTS_DIHEDRAL"
    
    def _plot_axes_modifier(self, ax, yonly=True):
        """
        Adjust plot axes to properly display dihedral angle values.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to modify.
        yonly : bool, optional
            Whether to apply modifications only to the y-axis, by default True.
        """
        return LabelsDihedralAnalyser._plot_axes_modifier(self, ax, yonly=yonly)

class GraphDatapointsDihedralAnalyser(GraphDatapointsAnalyser):
    """
    Data analyser for extracting and plotting dihedral angles from graph datasets.

    This converts geometric (cosine/sine) representations found in graph
    labels back into dihedral angles and plots them with appropriate axes limits.
    """

    _IDENTIFIER = "GRAPH_DATAPOINTS_DIHEDRAL"
    
    def _plot_axes_modifier(self, ax, yonly=True):
        """
        Adjust plot axes to properly display dihedral angle values.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to modify.
        yonly : bool, optional
            Whether to apply modifications only to the y-axis, by default True.
        """
        return LabelsDihedralAnalyser._plot_axes_modifier(self, ax, yonly=yonly)
    
    def _extract_labels(self, data):
        """
        Extract geometric labels from graph data and convert them to dihedral angles.

        Parameters
        ----------
        data : list
            Data points from the dataset.

        Returns
        -------
        dict
            Dictionary mapping label names to their converted angular values (in radians).
        """
        labels = super()._extract_labels(data)
        dihedrals = cos_sin_to_angle(labels)
        return dihedrals

