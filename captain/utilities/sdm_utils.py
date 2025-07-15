import os
import sys
import rioxarray as rxr
import warnings
import rasterio
import tifffile
import numpy as np
from scipy import ndimage
import glob
np.set_printoptions(suppress=True, precision=3)  # prints floats, no scientific notation
from ..agents.state_monitor import get_quadrant_indx_grid, get_sum_grid_value_quadrant

def z_transform(x):
 return (x - np.nanmean(x)) / np.nanstd(x)

def feature_transform(x, delta, denom):
    return (x - delta) / denom

def get_features_sdm(bathy, temp, oxygen,
                     future_temp=None,
                     future_oxygen=None,
                     include_coords=None, # xarray
                     reorder=None,
                     convolution_padding=10,
                     rescalers=None,
                     ):
    b = bathy.flatten()
    t = temp.flatten()
    o = oxygen.flatten()

    if reorder is None:
        reorder = np.arange(len(b))

    feat_list = [feature_transform(b, rescalers['bathy'][0], rescalers['bathy'][1])[reorder],
                 feature_transform(t, rescalers['temp'][0], rescalers['temp'][1])[reorder],
                 feature_transform(o, rescalers['oxygen'][0], rescalers['oxygen'][1])[reorder]]

    if future_temp is not None:
        t_fut = future_temp.flatten()
        o_fut = future_oxygen.flatten()
        feat_list_future = [feature_transform(b, rescalers['bathy'][0], rescalers['bathy'][1])[reorder],
                            feature_transform(t_fut, rescalers['temp'][0], rescalers['temp'][1])[reorder],
                            feature_transform(o_fut, rescalers['oxygen'][0], rescalers['oxygen'][1])[reorder]]

    if include_coords is not None:
        coords = np.meshgrid(include_coords[0]['x'].to_numpy(),
                               include_coords[0]['y'].to_numpy())
        l = coords[0].flatten()
        g = coords[1].flatten()

        feat_list = feat_list + [feature_transform(l, rescalers['lat'][0], rescalers['lat'][1])[reorder],
                                 feature_transform(g, rescalers['lon'][0], rescalers['lon'][1])[reorder]]
        if future_temp is not None:
            feat_list_future = feat_list_future + [
                feature_transform(l, rescalers['lat'][0], rescalers['lat'][1])[reorder],
                feature_transform(g, rescalers['lon'][0], rescalers['lon'][1])[reorder]]

    if convolution_padding > 1:
        k = np.ones((convolution_padding, convolution_padding)) / (convolution_padding ** 2)
        temp_tmp = temp[0].to_numpy() + 0
        temp_tmp[np.isfinite(temp_tmp) == False] = -10
        temp_conv = ndimage.convolve(temp_tmp, k, mode='reflect')
        temp_conv_z = feature_transform(temp_conv.flatten(), rescalers['temp'][0], rescalers['temp'][1])

        bathy_tmp = bathy[0].to_numpy() + 0
        bathy_conv = ndimage.convolve(bathy_tmp, k, mode='reflect')
        bathy_conv_z = feature_transform(bathy_conv.flatten(), rescalers['bathy'][0], rescalers['bathy'][1])
        feat_list = feat_list + [temp_conv_z[reorder],
                                 bathy_conv_z[reorder]]

        if future_temp is not None:
            f_temp_tmp = future_temp[0].to_numpy() + 0
            f_temp_tmp[np.isfinite(temp_tmp) == False] = -10
            f_temp_conv = ndimage.convolve(temp_tmp, k, mode='reflect')
            f_temp_conv = feature_transform(f_temp_conv.flatten(), rescalers['temp'][0], rescalers['temp'][1])
            feat_list_future = feat_list_future + [f_temp_conv[reorder],
                                                   bathy_conv_z[reorder]]


    features = np.array(feat_list).T
    if future_temp is not None:
        features_future = np.array(feat_list_future).T
    else:
        features_future = None
    return features, features_future


def load_map(filename):
    if ".tif" in filename:
        warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
        sp_data = rxr.open_rasterio(filename, masked=True, parse_coordinates=False)
        sp_data_cropped = sp_data.to_numpy()[0]
        taxon_name = os.path.basename(filename).split(".tif")[0]
    elif ".npz" in filename:
        sp_data = np.load(filename, allow_pickle=True)
        sp_data_cropped = sp_data['arr_0'].item()['x']
        taxon_name = os.path.basename(filename).split(".npz")[0]
    else:
        print("File not found", filename)
        sys.exit()
    return sp_data_cropped, taxon_name


def get_sdm_data(sdm_files, taxon_index, cropped=None):
    if ".tif" in sdm_files[taxon_index]:
        sp_data = rxr.open_rasterio(sdm_files[taxon_index], masked=True)
        if cropped is not None:
            sp_data_cropped = crop_data(sp_data, cropped)
        else:
            sp_data_cropped = sp_data.to_numpy()[0]
        taxon_name = os.path.basename(sdm_files[taxon_index]).split(".tif")[0]
    elif ".npz" in sdm_files[taxon_index]:
        sp_data = np.load(sdm_files[taxon_index], allow_pickle=True)
        sp_data_cropped = sp_data['arr_0'].item()['x']
        taxon_name = os.path.basename(sdm_files[taxon_index]).split(".npz")[0]
    elif ".npy" in sdm_files[taxon_index]:
        sp_data_cropped = np.load(sdm_files[taxon_index])
        taxon_name = os.path.basename(sdm_files[taxon_index]).split(".npy")[0]

    return sp_data_cropped, taxon_name

def crop_data(layer, cropped):
    layer_np = layer.to_numpy()[0][:, cropped]
    return layer_np


def get_data_from_list(sdm_dir, tag="/*.tif", max_species_cutoff=None,
                       rescale=False, zero_to_nan=False, global_rescale=False):
    sdm_files = np.sort(glob.glob(sdm_dir + tag))
    sdms = []
    sp_names = []
    for i in range(len(sdm_files)):
        sp_data, taxon_name = get_sdm_data(sdm_files, i, cropped=None)
        if rescale:
            sp_data = sp_data.astype(float) / np.nanmax(sp_data.astype(float))
        sdms.append(sp_data)
        sp_names.append(taxon_name)
        if i + 1 == max_species_cutoff:
            break

    sdms = np.squeeze(np.array(sdms))  # shape = (species, lon, lat)
    if global_rescale:
        sdms = sdms.astype(float) / np.nanmax(sdms.astype(float))

    if zero_to_nan:
        sdms[sdms == 0] = np.nan
    return sdms, sp_names


def get_graph(sdms, grid_length=None):
    species_richness = np.nansum(sdms, axis=0)
    original_grid_shape = species_richness.shape
    # MAKE IT A GRAPH without gaps
    reference_grid_pu = species_richness > 0
    reference_grid_pu_nan = reference_grid_pu.astype(float)
    reference_grid_pu_nan[reference_grid_pu_nan == 0] = np.nan
    # reduce coordinates
    xy_coords = np.meshgrid(np.arange(original_grid_shape[1]), np.arange(original_grid_shape[0]))
    graph_coords, _, __ = grid_to_graph(np.array(xy_coords), reference_grid_pu, grid_length=grid_length)
    return original_grid_shape, reference_grid_pu, reference_grid_pu_nan, xy_coords, graph_coords


def grid_to_graph(grid, reference_grid_pu, n_pus=None, nan_to_zero=False, grid_length=None,
                  return_dict=None, layer_name=None):
    if n_pus is None:
        n_pus = reference_grid_pu[reference_grid_pu > 0].size
    if grid_length is None:
        grid_length = np.round(np.sqrt(n_pus)).astype(int) + 1
    if grid_length % 2 != 0: # make it a multiple of 2
        grid_length += 1

    if len(grid.shape) == 3:
        m_graph = []
        for scaled_dd in grid:
            prep_vec = np.zeros(grid_length ** 2)
            species_vec = scaled_dd[reference_grid_pu > 0].flatten()
            prep_vec[:n_pus] += species_vec
            species_grid = prep_vec.reshape((grid_length, grid_length))
            species_grid[np.isnan(species_grid)] = 0
            m_graph.append(species_grid)

        m_graph = np.array(m_graph)

    elif len(grid.shape) == 2:
        m_graph = np.zeros(grid_length ** 2)
        m_graph[:n_pus] += grid[reference_grid_pu > 0].flatten()
        m_graph = m_graph.reshape((grid_length, grid_length))

    else:
        m_graph = None
        print("ERROR: Data not recognized")

    if nan_to_zero:
        m_graph[np.isnan(m_graph)] = 0

    if return_dict is not None:
        if layer_name is None:
            if len(grid.shape) == 3:
                layer_name = ["dat_%s" % i for i in range(m_graph.shape[0])]
                graph_dict = dict(zip(layer_name, list(m_graph)))
            else:
                graph_dict = {"dat_0": m_graph}
        else:
            graph_dict = dict(zip(layer_name, list(m_graph)))

        m_graph = graph_dict

    return m_graph, n_pus, grid_length

def graph_to_grid(graph, reference_grid_pu, n_pus=None, zero_to_nan=False,
                  return_dict=None, layer_name=None):
    if n_pus is None:
        n_pus = reference_grid_pu[reference_grid_pu > 0].size

    original_grid_shape = reference_grid_pu.shape

    if len(graph.shape) == 3:
        m_grid = []
        for var in graph:
            tmp = var.flatten()[:-(var.size - n_pus)] + 0
            z = np.zeros(original_grid_shape)
            z[reference_grid_pu > 0] += tmp
            if zero_to_nan:
                z[reference_grid_pu == 0] = np.nan
            m_grid.append(z)

        m_grid = np.array(m_grid)
    else:
        tmp = graph.flatten()[:-(graph.size - n_pus)] + 0
        m_grid = np.zeros(original_grid_shape)
        m_grid[reference_grid_pu > 0] += tmp
        if zero_to_nan:
            m_grid[reference_grid_pu == 0] = np.nan

    if return_dict is not None:
        if layer_name is None:
            if len(graph.shape) == 3:
                layer_name = ["dat_%s" % i for i in range(m_grid.shape[0])]
                graph_dict = dict(zip(layer_name, list(m_grid)))
            else:
                graph_dict = {"dat_0": m_grid}
        else:
            graph_dict = dict(zip(layer_name, list(m_grid)))

        m_grid = graph_dict

    return m_grid



def npz_to_tiff(npz_file, tiff_file):
    """
    Reads data from an .npz file and saves it as a .tiff file.

    Args:
        npz_file (str): Path to the input .npz file.
        tiff_file (str): Path to the output .tiff file.
    """
    try:
        data = np.load(npz_file, allow_pickle=True)
        # data = np.load(npz_file)
        # Assuming the .npz file contains a single array.
        # If it contains multiple arrays, you might need to specify which one to save.
        if len(data.files) == 1:
            array_name = data.files[0]
            image_data = data['arr_0'].item()['x'].astype(np.float32)
            print(image_data.shape, np.min(image_data), np.max(image_data), image_data.dtype)
            tifffile.imwrite(tiff_file, image_data)
            print(f"Converted '{npz_file}' to '{tiff_file}'")
        elif len(data.files) > 1:
            print(f"Warning: '{npz_file}' contains multiple arrays. Saving the first one found.")
            array_name = data.files[0]
            image_data = data['arr_0'].item()['x']
            tifffile.imwrite(tiff_file, image_data)
            print(f"Converted the first array from '{npz_file}' to '{tiff_file}'")
        else:
            print(f"Error: '{npz_file}' contains no arrays.")
    except FileNotFoundError:
        print(f"Error: NPZ file not found: '{npz_file}'")
    except Exception as e:
        print(f"An error occurred while processing '{npz_file}': {e}")

def convert_npz_folder_to_tiff(npz_folder, tiff_folder=None):
    """
    Finds all .npz files in a folder and converts them to .tiff files.

    Args:
        npz_folder (str): Path to the folder containing the .npz files.
        tiff_folder (str, optional): Path to the folder where the .tiff files
                                     will be saved. If None, they are saved
                                     in the same directory as the .npz files.
                                     Defaults to None.
    """
    if not os.path.isdir(npz_folder):
        print(f"Error: NPZ folder not found: '{npz_folder}'")
        return

    if tiff_folder is not None and not os.path.exists(tiff_folder):
        os.makedirs(tiff_folder)

    npz_files = glob.glob(os.path.join(npz_folder, "*.npz"))

    if not npz_files:
        print(f"No .npz files found in '{npz_folder}'.")
        return

    for npz_file in npz_files:
        base_name = os.path.splitext(os.path.basename(npz_file))[0]
        if tiff_folder:
            tiff_file = os.path.join(tiff_folder, f"{base_name}.tif")
        else:
            tiff_file = os.path.join(os.path.dirname(npz_file), f"{base_name}.tif")
        npz_to_tiff(npz_file, tiff_file)



def array_to_tiff(dat, file_names=None, wd_output="", by_layer=True):
    try:
        os.makedirs(wd_output)
    except FileExistsError:
        pass
    if by_layer:
        if file_names  is None:
            file_names = ["data_%s.tif" % i for i in range(dat.shape[0])]
        for i in range(dat.shape[0]):
            image_data = dat[i]
            tifffile.imwrite(os.path.join(wd_output, file_names[i]), image_data)

    else:
        tifffile.imwrite(os.path.join(wd_output, file_names), dat)


def rank_cells(layer_dict,
               numerator=None, # multiply variables
               denominator=None, # divide
               num_weights=None,
               den_weights=None,
               min_den=0.1, # minimum denominator
               q=None, # quantile (returns 1 for the top q scoring cells, zero for the others
               rescale=True, # rescale between 0 and 1
               func=np.prod
               ):
    if numerator is not None:
        if num_weights is None:
            num_weights = np.ones(len(numerator))
        num = np.array([layer_dict[numerator[l]] ** num_weights[l] for l in range(len(numerator))])
    else:
        num = 1
    if denominator is not None:
        if den_weights is None:
            den_weights = np.ones(len(denominator))
        den = np.array([layer_dict[denominator[l]] ** den_weights[l] for l in range(len(denominator))])
        den[den < min_den] = min_den
    else:
        den = 1

    score = func(num, 0) / func(den, 0)
    if q is not None and q > 0:
        q30 = np.quantile(score, q=(1 - q))
        relative_reward = score + 0
        print(relative_reward[relative_reward > q30].size / relative_reward.size)
        relative_reward[relative_reward < q30] = 0
        relative_reward[relative_reward > 0] = 1.
        score = relative_reward + 0
    if rescale:
        score /= np.max(score)
    return score






















