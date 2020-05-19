import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat

from sklearn.decomposition import PCA

from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
import matplotlib.patheffects as pe

def preprocess(datas,
               times,
               tstart=-50,
               tend=150,
               soft_normalize=5,
               subtract_cc_mean=True,
               pca=True,
               num_pcs=6):
        """
        Preprocess data for jPCA. 
        
        Args
        ----
            datas: List of trials, where each element of the list has shape Times x Neurons. 
                As an example, this might be the output of load_churchland_data()
                
            times: List of times for the experiment. Typically, time zero corresponds
                to a stimulus onset. This list is used for extracting the set of
                data to be analyzed (see tstart and tend args).
                
            tstart: Integer. Starting time for analysis. For example, if times is [-10, 0 , 10]
                    and tstart=0, then the data returned by this function will start
                    at index 1.
                    
            tend: Integer. Ending time for analysis.
            
            soft_normalize: Float or Int. Constant used during soft-normalization preprocessing step.
                            Adapted from original jPCA matlab code. Normalized firing rate is 
                            computed by dividing by the range of the unit across all conditions and times
                            plus the soft_normalize constant: Y_{cij} = (max(Y_{:i:}) - min(Y_{:i:}) + C)
                            where Y_{cij} is the cth condition, ith neuron, at the jth time bin.
                            C is the constant provided by soft_normalize. Set C negative to skip the 
                            soft-normalizing step.
                            
            subtract_cc_mean: Boolean. Whether or not to subtract the mean across conditions. Default True.
            
            pca: Boolean. True to perform PCA as a preprocessing step. Defaults to True.
            
            num_pcs: Int. When pca=True, controls the number of PCs to use. Defaults to 6.
            
        Returns
        -------
            data_list: List of arrays, each T x N. T will depend on the values
                passed for tstart and tend. N will be equal to num_pcs if pca=True.
                Otherwise the dimension of the data will remain unchanged.
            full_data_variance: Float, variance of original dataset.
            pca_variance_captured: Array, variance captured by each PC. 
                                   If pca=False, this is set to None.
        """
        datas = np.stack(datas)
        num_conditions, num_time_bins, num_units = datas.shape

        if soft_normalize > 0:
            fr_range = np.max(datas, axis=(0,1)) - np.min(datas, axis=(0,1))
            datas /= (fr_range + soft_normalize)
            
        if subtract_cc_mean:
            cc_mean = np.mean(datas, axis=0)
            datas -= cc_mean

        # For consistency with the original jPCA matlab code,
        # we compute PCA using only the analyzed times.
        idx_start = times.index(tstart)
        idx_end = times.index(tend) + 1 # Add one so idx is inclusive
        datas = datas[:, idx_start:idx_end, :]
        num_time_bins = idx_end - idx_start
        
        # Reshape to perform PCA on all trials at once.
        X_full = datas.reshape(num_time_bins * num_conditions, num_units)
        full_data_var = np.sum(np.diag(np.cov(X_full.T)))
        pca_variance_captured = None

        if pca:
            pca = PCA(num_pcs)
            datas = pca.fit_transform(X_full)
            datas = datas.reshape(num_conditions, num_time_bins, num_pcs)
            pca_variance_captured = pca.explained_variance_

        data_list = [x for x in datas]
        return data_list, full_data_var, pca_variance_captured

def plot_trajectory(ax, x, y, 
                    color="black",
                    outline="black",
                    circle=True,
                    arrow=True):
    """
    Plot a single neural trajectory in a 2D plane.
    
    Args
    ----
        ax: Axis used for plotting.
        
        x: Values of variable on x-axis.
        
        y: Values of variable on y-axis.
        
        color: Fill color of line to be plotted. Defaults to "black".
        
        outline: Outline color of line. Defaults to "black".
        
        circle: True if the trajectory should have a circle at its starting state.
        
        arrow: True if the trajectory should have an arrow at its ending state.
        
    """
    ax.plot(x, y,
             color=color,
             path_effects=[pe.Stroke(linewidth=2, foreground=outline), pe.Normal()])
        
    if circle:
        circ = plt.Circle((x[0], y[0]),
                          radius=0.05, 
                          facecolor=color,
                          edgecolor="black")
        ax.add_artist(circ)

    if arrow:
        dx = x[-1] - x[-2]
        dy = y[-1] - y[-2]
        px, py = (x[-1], y[-1])
        ax.arrow(px, py , dx, dy, 
                  facecolor=color, 
                  edgecolor=outline,
                  length_includes_head=True,
                  head_width=0.05)

def plot_projections(data_list,
                     x_idx=0,
                     y_idx=1,
                     arrows=True,
                     circles=True,
                     align_prep_state=True):
    """
    Plot trajectories found via jPCA or PCA. 
    
    Args
    ----
        data_list: List of trajectories, where each entry of data_list is an array of size T x D, 
                   where T is the number of time-steps and D is the dimension of the projection.
        x_idx: column of data which will be plotted on x axis. Default 0.
        y_idx: column of data which will be plotted on y axis. Default 0.
        arrows: True to add arrows to the trajectory plot.
        circles: True to add circles at the beginning of each trajectory.
        sort_colors: True to color trajectories based on the starting x coordinate. This mimics
                     the jPCA matlab toolbox.
    """
    fig = plt.figure(figsize=(5,5))
    colormap = plt.cm.RdBu
    colors = np.array([colormap(i) for i in np.linspace(0, 1, len(data_list))])
    data_list = [data[:, [x_idx, y_idx]] for data in data_list]
    basis = np.array([[1, 0], [0,1]])
    
    # # Optional: align prep states so that they spread along the horizontal axis.

    prep_states = np.row_stack([data[0] for data in data_list])
    pca = PCA(n_components=2)
    pca = pca.fit(prep_states)
    rot = pca.components_

    pc1 = np.append(rot[:,0], 0)
    pc2 = np.append(rot[:, 1], 0)
    cross = np.cross(pc1, pc2)
    if cross[2] < 0:
        rot[:, 1] = -rot[:, 1]
    basis = basis @ rot

    test_proj = np.concatenate(data_list) @ basis
    if np.max(np.abs(test_proj[:, 1])) > np.max(test_proj[:, 1]):
        basis = -basis

    data_list = data_list @ basis
    # Sort colors left to right bases on prep states position
    # on horizontal axis.
    start_x_list = [data[0,0] for data in data_list]
    color_indices = np.argsort(start_x_list)

    ax = plt.gca()
    for i, data in enumerate(np.array(data_list)[color_indices]):
        plot_trajectory(ax,
                        data[:, 0],
                        data[:, 1],
                        color=colors[i],
                        circle=circles,
                        arrow=arrows)


def load_churchland_data(path):
    """
    Load data from Churchland, Cunningham et al, Nature 2012
    Data available here: 
    https://churchland.zuckermaninstitute.columbia.edu/content/code
    
    Returns a 3D data array, C x T x N. T is the number of time bins,
    and N is the number of neurons (218), and C is the number of conditions.

    Note: Loading a .mat struct in Python is quite messy beause the formatting is messed up,
    which is why we need to do some funky indexing here.
    """
    struct = loadmat(path)
    conditions = struct["Data"][0]

    # For this dataset, times are the same for all conditions,
    # but they are formatted strangely -- each element of the times
    # vector is in a separate list.
    datas = None
    times = [t[0] for t in conditions[0][1]]
    
    for cond in conditions:
        spikes = cond[0]
        if datas is None:
            datas = spikes
        else:
            datas = np.dstack((datas, spikes))
    
    datas = np.moveaxis(datas, 2, 0)
    datas = [x for x in datas]
    return datas, times

def ensure_datas_is_list(f):
    def wrapper(self, datas, **kwargs):
        datas = [datas] if not isinstance(datas, list) else datas
        return f(self, datas, **kwargs)
    return wrapper

