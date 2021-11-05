import numpy as np
import os
import sys

import netCDF4 as nc4

from src.track_alignment import map_and_reduce
from src.utils import get_datetime, get_file_time_info, minutes_since    

swath_channels = ['ev_250_aggr1km_refsb_1', 'ev_250_aggr1km_refsb_2', 'ev_500_aggr1km_refsb_3', 'ev_500_aggr1km_refsb_4', 'ev_500_aggr1km_refsb_5', 'ev_500_aggr1km_refsb_6', 'ev_500_aggr1km_refsb_7', 'ev_1km_refsb_8', 'ev_1km_refsb_9', 'ev_1km_refsb_10', 'ev_1km_refsb_11', 'ev_1km_refsb_12', 'ev_1km_refsb_13L', 'ev_1km_refsb_13H', 'ev_1km_refsb_14L', 'ev_1km_refsb_14H', 'ev_1km_refsb_15', 'ev_1km_refsb_16', 'ev_1km_refsb_17', 'ev_1km_refsb_18', 'ev_1km_refsb_19',   'ev_1km_emissive_20', 'ev_1km_emissive_21', 'ev_1km_emissive_22', 'ev_1km_emissive_23', 'ev_1km_emissive_24', 'ev_1km_emissive_25', 'ev_1km_refsb_26', 'ev_1km_emissive_27', 'ev_1km_emissive_28', 'ev_1km_emissive_29', 'ev_1km_emissive_30', 'ev_1km_emissive_31', 'ev_1km_emissive_32', 'ev_1km_emissive_33', 'ev_1km_emissive_34', 'ev_1km_emissive_35', 'ev_1km_emissive_36', 'latitude', 'longitude','solar_zenith_angle','cloud_mask']

layer_info_channels = {
    "CloudLayerType" : 'cloud_layer_type'
}

def copy_dataset_structure(original_filename, copy_filename, deep=True, zlib=True):
    
    with nc4.Dataset(original_filename, 'r') as original:

        copy = nc4.Dataset(copy_filename, 'w', format='NETCDF4')

        block_list = [(copy, original)]
        variables = {}

        # create groups if deep
        if deep:
            for name, group in original.groups.items():
                new_group = copy.createGroup(name)
                block_list.append((new_group, group))

        # copy global attributes
        copy.setncatts({a : original.getncattr(a) for a in original.ncattrs()})

        for new_block, block in block_list:

            # copy dimensions
            for name, dim in block.dimensions.items():
                new_dim = new_block.createDimension(name, len(dim) if not dim.isunlimited() else None)

            # Copy variables
            for name, var in block.variables.items():

                new_var = new_block.createVariable(name, var.datatype, var.dimensions, zlib=zlib)
                
                # Copy variable attributes
                new_var.setncatts({a : var.getncattr(a) for a in var.ncattrs()})

                variables[name] = new_var

    return copy, variables

def fill_dataset(dataset, variables, swath, layer_info, minutes, status="daylight", deep=True):

    shape = swath[0].shape

    for i, channel in enumerate(swath_channels):

        variables[channel][0] = swath[i].T

    if layer_info is not None:

        for info_name, channel in layer_info_channels.items():

            # map data to swath format
            if info_name == "PrecipFlag":
                # precipitation flag is not available per layer
                info = np.full(shape, variables[channel]._FillValue)
                map_and_reduce(layer_info["mapping"], layer_info[info_name], info, layer_info["width-range"])
                info = info.T

            else:

                info = np.full((*shape, 10), variables[channel]._FillValue)

                # correct values, from [1, 8] to [0, 7]
                if info_name == "CloudLayerType":
                    info -= 1
                    info[info < 0] = variables[channel]._FillValue

            variables[channel][0] = info

    # set global variables and attributes
    dataset.status_flag = status
    dataset["time"][0] = minutes

def load_npys(swath_path, layer_info_dir="layer-info"):

    dirname, filename = os.path.split(swath_path)

    swath = np.load(swath_path)

    try:
        layer_info_dict = np.load(os.path.join(dirname, layer_info_dir, filename)).item()

    except FileNotFoundError:
        layer_info_dict = None

    return swath, layer_info_dict

def save_as_nc(swath, layer_info, swath_path, save_name):

    copy, variables = copy_dataset_structure(os.path.join("netcdf","datasetstr.nc"), save_name)

    # determine swath status from directory hierarchy
    status = "corrupt"
    if "daylight" in save_name:
        status = "daylight"
    elif "night" in save_name:
        status = "night"

    # convert npy to nc
    year, abs_day, hour, minute = get_file_time_info(swath_path)
    #month = get_datetime(year, int(abs_day)).month
    minutes_since_2016 = minutes_since(int(year), int(abs_day), int(hour), int(minute))
    fill_dataset(copy, variables, swath, layer_info, minutes_since_2016, status)

    copy.close()

if __name__ == "__main__":

    swath_path = sys.argv[2]
    save_dir = sys.argv[1]

    swath, layer_info = load_npys(swath_path)

    # get time info
    year, abs_day, hour, minute = get_file_time_info(swath_path)
    month = get_datetime(year, int(abs_day)).month

    # determine swath status from directory hierarchy
    status = "corrupt"
    if "daylight" in swath_path:
        status = "daylight"
    elif "night" in swath_path:
        status = "night"

    # create save directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #create a copy of reference dataset
    copy_name = "A{}.{}.{}{}.nc".format(year, abs_day, hour, minute)
    copy, variables = copy_dataset_structure(os.path.join("netcdf", "cumulo.nc"), os.path.join(save_dir, month, status, copy_name))

    # convert npy to nc
    minutes_since_2008 = minutes_since(int(year), int(abs_day), int(hour), int(minute))
    fill_dataset(copy, variables, swath, layer_info, minutes_since_2008, status)

    copy.close()
