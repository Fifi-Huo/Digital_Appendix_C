import numpy as np
import os
import time
import traceback

from PIL import Image
from datetime import date

import src.cloudsat
import src.interpolation
import src.modis_level1
import src.modis_level2


def extract_swath_ontrack(myd02_filename, myd03_dir, myd35_dir, cloudsat_lidar_dir, cloudsat_dir, save_dir, verbose=0, save=True):
    """
    :param myd02_filename: the filepath of the radiance (MYD02) input file
    :param myd03_dir: the root directory of geolocational (MYD03) files
    :param myd35_dir: the root directory to cloud mask files
    :param cloudsat_lidar_dir: the root directory of cloudsat-lidar files
    :param cloudsat_dir: the root directory of cloudsat files
    :param save_dir:
    :param verbose: verbosity switch: 0 - silent, 1 - verbose, 2 - partial, only prints confirmation at end
    :return: none
    Expects to find a corresponding MYD03 file in the same directory. Comments throughout
    """

    tail = os.path.basename(myd02_filename)

    # creating the save directories
    save_dir_daylight = os.path.join(save_dir, "daylight")
    save_dir_night = os.path.join(save_dir, "night")
    #save_dir_test = os.path.join(save_dir, "test")
    save_dir_corrupt = os.path.join(save_dir, "corrupt")

    for dr in [save_dir_daylight, save_dir_night, save_dir_corrupt]:
        if not os.path.exists(dr):
            os.makedirs(dr)

    # pull a numpy array from the hdfs
    np_swath = src.modis_level1.get_swath(myd02_filename, myd03_dir)

    if verbose:
        print("swath {} loaded".format(tail))

    # as some bands have artefacts, we need to interpolate the missing data - time intensive
    t1 = time.time()
    
    filled_ch_idx = src.interpolation.fill_all_channels(np_swath)  
    
    t2 = time.time()

    if verbose:
        print("Interpolation took {} s".format(t2-t1))
        print("Channels", filled_ch_idx, "are now full")
        print("length of channels",len(filled_ch_idx))

    # if all channels were filled
    if len(filled_ch_idx) == 41:
        save_subdir = save_dir_daylight

    # if all but visible channels were filled
    elif len(filled_ch_idx) == 19:
        save_subdir = save_dir_night

    else:
        save_subdir = save_dir_corrupt

    # pull cloud mask channel
    cm = src.modis_level2.get_cloud_mask(myd02_filename, myd35_dir)

    if verbose:
        print("Cloud mask loaded")

    # get cloudsat alignment - time intensive
    t1 = time.time()

    try:
        # alignment returns:
        # cs_range: minimal and maximal column indices of the satellite track for the current swath 
        # mapping: cloudsat-pixels -> swath pixels
        # laye_info: available cloudsat variable values for the current swath
        cs_range, mapping, mapping_a, mapping_b, layer_info = src.cloudsat.get_cloudsat_mask(myd02_filename, cloudsat_lidar_dir, cloudsat_dir, np_swath[-2], np_swath[-1], map_label=False)

    except Exception as e:
        print("Couldn't extract cloudsat track of {}: {}".format(tail, e))
        traceback.print_exc(file=sys.stdout)
    else:
        print("Nothing went wrong")

    t2 = time.time()

    if verbose:
        print("Cloudsat alignment took {} s".format(t2 - t1))

    # cast swath values in the range of the satellite track, cast swath values to float
    np_swath = np.vstack([np_swath, cm]).astype(np.float32)
    print(np_swath.shape, np_swath[0].shape, np_swath[0].shape[1])
    
    index = [np_swath[0].shape[1] - i for i in mapping_b]
    np_swath_final = np.zeros((66, mapping.shape[0]))
    print(np_swath_final.shape, np_swath_final[0].shape)
    
    for cl in range (np_swath.shape[0]):
        temp = np_swath[cl]
        c = 0
        for i in mapping_a:
            reshape_v = temp[i]
            #print(reshape_v.shape)
            np_swath_final[cl][c] = reshape_v[index[c]]
            c = c+1

    # create the save path for the swath array, and save the array as a npy, with the same name as the input file.
    swath_savepath_str = os.path.join(save_subdir, tail.replace(".hdf", ".npy"))
    
    if verbose:
        print("Swath")
    
    if save:
        np.save(swath_savepath_str, np_swath, allow_pickle=False)

        if verbose:
            print("swath saved as {}".format(swath_savepath_str))
    
    try:

        #layer_info.update({"width-range": cs_range, "mapping": mapping})
        
        if save:

            layer_info_savepath = os.path.join(save_subdir, "layer-info")
    
            if not os.path.exists(layer_info_savepath):
                os.makedirs(layer_info_savepath)
            
            np.save(os.path.join(layer_info_savepath, tail.replace(".hdf", ".npy")), layer_info)
            if verbose:
                print("layer-info saved as {}".format(layer_info_savepath))

    except:
        
        layer_info = None

    return np_swath_final, layer_info, save_subdir, tail



# Hook for bash
if __name__ == "__main__":

    import sys

    from pathlib import Path
    
    from netcdf.npy_to_nc import save_as_nc
    from src.utils import get_file_time_info
    
    #get the myd02 file directory (Channel 1-36, Swath pixel)
    myd02_filename = sys.argv[1]
    #save_dir = sys.argv[2]
    save_dir = '/Users/documents/Desk/Antarctic_sea_ice/Dataset_test/'
    
    root_dir, filename = os.path.split(myd02_filename)

    month, day = root_dir.split("/")[-2:]

    # get time info
    year, abs_day, hour, minute = get_file_time_info(myd02_filename)
    save_name = "A{}.{}.{}{}.nc".format(year, abs_day, hour, minute)

    # recursvely check if file exist in save_dir
    for _ in Path(save_dir).rglob(save_name):
        raise FileExistsError("{} already exist. Not extracting it again.".format(save_name))
    
    root_dir2 = "/Users/documents/Desk/Antarctic_sea_ice/Dataset_test/"
    
    #myd03 gec(lat,lon)
    myd03_dir = os.path.join(root_dir2, "MODIS", "MYD03", year, month, day)
    #return a mask, with 0 for cloudy, 1 for uncertain/probably cloudy, 2 for probably clear, and 3 for clear.
    myd35_dir = os.path.join(root_dir2, "MODIS", "MYD35_L2",  year, month, day)
    cloudsat_dir = None
    #label (cloud_occurrence) 0 -- non cloud determined or error, 1 -- cloud occurrences
    cloudsat_lidar_dir = os.path.join(root_dir2, "CloudSat", "2B-CLDCLASS-LIDAR", year, month)

    #cloudsat_dir = os.path.join(root_dir, "CloudSat")

    # extract training channels, validation channels, cloud mask, class occurences if provided
    np_swath, layer_info, save_subdir, swath_name = extract_swath_ontrack(myd02_filename, myd03_dir, myd35_dir, cloudsat_dir, cloudsat_lidar_dir, save_dir=save_dir, verbose=2, save=False)
    #np_swath: np-array co-located swath for (myd02,myd03,myd35)
    
  
    #save swath as netcdf
    test_name = os.path.join(save_subdir, save_name)
    save_as_nc(np_swath, layer_info, swath_name, test_name)
    
    if "corrupt" in save_subdir:
        print("Failed to extract tiles: tiles are extracted only from swaths with fully interpolated non-visible channels")
        exit(0)

    # # save visible channels as png for visualization purposes
    # extract_swath_rbg(myd02_filename, os.path.join(year, month, day), save_subdir, verbose=1)