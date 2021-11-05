import glob
import numpy as np
import os
import pickle

from pyhdf.SD import SD, SDC 
from pyhdf.HDF import HDF
from pyhdf.VS import VS

from src.track_alignment import get_track_oi, find_track_range, map_labels
from src.utils import get_datetime, get_file_time_info

def find_cloudsat_by_day(abs_day, year, cloudsat_lidar_dir):
    """ returns list of filenames of specified day, and of previous and following day """

    cloudsat_filenames = []

    for i in range(-1, 2):
        
        cur_day = abs_day + i

        pattern = "{}{}/{}{}{}*.hdf".format("0" * (3 - len(str(cur_day))), cur_day, year, "0" * (3 - len(str(cur_day))), cur_day)
        cloudsat_filenames += glob.glob(os.path.join(cloudsat_lidar_dir, pattern))

    return cloudsat_filenames

def find_matching_cloudsat_files(radiance_filename, cloudsat_lidar_dir):
    """
    :param radiance_filename: the filename for the radiance .hdf, demarcated with "MYD02".
    :return cloudsat_filenames: a list of paths to the corresponding cloudsat files (1 or 2 files)
    The time of the radiance file is used for selecting the cloudsat files: a MODIS swath is acquired every 5 minutes, while a CLOUDSAT granule is acquired every ~99 minutes. It can happen that a swath crosses over two granules. The filenames specify the starting time of the acquisition.
    CLOUDSAT filenames are in the format: AAAADDDHHMMSS_*.hdf
    """

    basename = os.path.basename(radiance_filename)

    year, abs_day, hour, minutes = get_file_time_info(basename)
    year, abs_day, hour, minutes = int(year), int(abs_day), int(hour), int(minutes)

    swath_dt = get_datetime(year, abs_day, hour, minutes)

    cloudsat_filenames = find_cloudsat_by_day(abs_day, year, cloudsat_lidar_dir)

    # collect all granules before and after swath's time
    prev_candidates, foll_candidates = {}, {}

    for filename in cloudsat_filenames:
        
        cs_time_info = os.path.basename(filename)
        year, day, hour, minute, second = int(cs_time_info[:4]), int(cs_time_info[4:7]), int(cs_time_info[7:9]), int(cs_time_info[9:11]), int(cs_time_info[11:13])

        granule_dt = get_datetime(year, day, hour, minute, second)

        if granule_dt <= swath_dt and (swath_dt - granule_dt).total_seconds() < 6000:
            prev_candidates[granule_dt] = filename

        elif granule_dt >= swath_dt and (granule_dt - swath_dt).total_seconds() < 300:
            foll_candidates[granule_dt] = filename

    prev_dt = max(prev_candidates.keys())
    
    # if swath crosses over two cloudsat granules, return both
    if len(foll_candidates.keys()) > 0:
        
        foll_dt = min(foll_candidates.keys())
        
        return prev_candidates[prev_dt], foll_candidates[foll_dt]
            
    return [prev_candidates[prev_dt]] 


def get_coordinates(cloudsat_filenames, verbose=0):
    
    all_latitudes, all_longitudes = [], []

    for cloudsat_path in cloudsat_filenames:
                
        f = HDF(cloudsat_path, SDC.READ) 
        vs = f.vstart() 
        
        vdata_lat = vs.attach('Latitude')
        vdata_long = vs.attach('Longitude')

        latitudes = vdata_lat[:]
        longitudes = vdata_long[:]
        
        assert len(latitudes) == len(longitudes), "cloudsat hdf corrupted"
        
        if verbose:
            print("hdf information", vs.vdatainfo())
            print('Nb pixels: ', len(latitudes))
            print('Lat min, Lat max: ', min(latitudes), max(latitudes))
            print('Long min, Long max: ', min(longitudes), max(longitudes))

        all_latitudes += latitudes
        all_longitudes += longitudes

        # close everything
        vdata_lat.detach()
        vdata_long.detach()
        vs.end()
        f.close()
    
    return np.array(all_latitudes), np.array(all_longitudes)


def get_layer_information(cloudsat_filenames, get_quality=True, verbose=0):
    """ Returns
    CloudLayerType: -9: error, 0: non determined, 1-8 cloud types 
    CloudLayerBase: in km
    CloudLayerTop: in km
    """
    

    for cloudsat_path in cloudsat_filenames:

        sd = SD(cloudsat_path, SDC.READ)
        
        if verbose:
            # List available SDS datasets.
            print("hdf datasets:", sd.datasets())
        
        # get cloud types at each height
        layer_info = np.array(sd.select('CloudLayerType').get())
        CloudLayerBase = np.array(sd.select('CloudLayerBase').get())
        CloudLayerTop = np.array(sd.select('CloudLayerTop').get())
        
        occurrences = np.zeros((layer_info.shape[0], 1))
        print(CloudLayerBase.shape[0], CloudLayerBase.shape[1])
        occ = 0 #occurrences[j]
        
        for i in range(CloudLayerBase.shape[0]):
            #occ = 0 #occurrences[j]
            nb_cloud_layer = np.where(CloudLayerBase[i,:] < 0 )[0][0]
            #print(nb_cloud_layer)
            for j in range(nb_cloud_layer):
            
                if CloudLayerBase.data[i,j] > 0 and CloudLayerTop.data[i,j] > 0.0:
                    nb_cloud_height = CloudLayerTop.data[i,j] - CloudLayerBase.data[i,j]
                    #print(nb_cloud_height)
                
                    if nb_cloud_height > 0:
                        occ = 1 
            
            occurrences[i] = occ


    return occurrences


def get_class_occurrences(layer_types):
    """ 
    Takes in a numpy.ndarray of size (nb_points, 10) describing for each point of the track the types of clouds identified at each of the 10 heights 
    counting the number of times 8 type of clouds was spotted vertically.
    and returns occrrences (binary) as the label of the present/absent of cloud
    
    The height and cloud type information is then lost. 
    """
    
    occurrences = np.zeros((layer_types.shape[0], 1))
    
    
    for i in range(layer_types.shape[0]):
        occ = 0 #occurrences[i]
        g = 0
        labels  = layer_types[i]
        
        for l in labels:
                
            # keep only cloud types (no 0 or -9)
            if l > 0:
                occ = 1 
            g += l
            
            if g == -90:
                occ = -1
            
        occurrences[i] = occ   
               
    return occurrences    

def get_cloudsat_mask(l1_filename, cloudsat_lidar_dir, cloudsat_dir, swath_latitudes, swath_longitudes, map_label=True):

    # retrieve cloudsat files content
    if cloudsat_lidar_dir is None:

        cloudsat_filenames = find_matching_cloudsat_files(l1_filename, cloudsat_dir)
        # LayerTypeQuality not available in CS_2B-CLDCLASS_GRANULE_P1_R05_E02_F00 files
        layer_info = get_layer_information(cloudsat_filenames, get_quality=False) 

    else:

        cloudsat_filenames = find_matching_cloudsat_files(l1_filename, cloudsat_lidar_dir)

            
        layer_info = get_layer_information(cloudsat_filenames, get_quality=True)
        #print("layerinfo", layer_info)

    # focus around cloudsat track
    cs_latitudes, cs_longitudes = get_coordinates(cloudsat_filenames)
 
    cs_range, mapping, mapping_a, mapping_b, mapping_c = get_track_oi(cs_latitudes, cs_longitudes, swath_latitudes, swath_longitudes)
    #cs_latitudes, cs_longitudes = cs_latitudes[toi_indices], cs_longitudes[toi_indices]
    #lat, lon = swath_latitudes[:, cs_range[0]:cs_range[1]], swath_longitudes[:, cs_range[0]:cs_range[1]]
   
    
    layer_info = layer_info[mapping_c]
    
    zero_inds = np.where(layer_info == 0)[0]
    non_zero_inds = np.where(layer_info > 0)[0]
    
    inds = np.append(zero_inds[:len(non_zero_inds)],non_zero_inds)
    print(inds.shape, zero_inds.shape, non_zero_inds.shape)
    #cloud_occurrences = get_class_occurrences(layer_info)
    



    if map_label:
        #cloud_occurrences = get_class_occurrences(layer_info)
        cloudsat_mask = map_labels(mapping, cloud_occurrences, lat.shape)
        #print(cloudsat_mask.shape)

        # remove labels on egdes
        #cloudsat_mask[:10] = 0
        #cloudsat_mask[:-11:-1] = 0

        #print("retrieved", np.sum(cloudsat_mask > 0), "labels")
        
        # go back to initial swath size
        ext_cloudsat_mask = np.zeros((*(lat.shape), 1))
        #print(ext_cloudsat_mask.shape)
        ext_cloudsat_mask = cloudsat_mask
        cloudsat_result = ext_cloudsat_mask.transpose(2, 0, 1).astype(np.uint8)
        
        #print(ext_cloudsat_mask.shape, cloudsat_result.shape)

        return cs_range, mapping, mapping_a, mapping_b, cloudsat_result
    
    else:

        return cs_range, mapping, mapping_a, mapping_b, layer_info
    
    
