import numpy as np
import random

from scipy.stats import mode
from sklearn.metrics.pairwise import manhattan_distances
from scipy.spatial import distance_matrix
from scipy.spatial.distance import pdist, cdist

MAX_WIDTH, MAX_HEIGHT = 1354, 2040


def get_track_oi(cs_latitudes, cs_longitudes, swath_latitudes, swath_longitudes):
    
    """
    :param cs_lat, cs_lon:latitudes and longitudes from cloudsat track 
    :param swath_lat, swath_lon:latitudes and longitudes from MODIS swath in the range of the cloudsat track
    :return data: index of track, cloudsat-pixels -> swath pixels 
    """

    modis_granule_shape = swath_latitudes.shape
    modis_dim_along_track = modis_granule_shape[0] - 1
    
    # Find First Colocated Pixel

    modis_pt1 = np.stack((swath_latitudes[0,:],swath_longitudes[0,:]), axis=-1)

    cloudsat_pt1 = np.stack((cs_latitudes[:,0],cs_longitudes[:,0]), axis=-1)

    d1 = cdist(cloudsat_pt1,modis_pt1)

    d1_min = np.argwhere(d1==d1.min())

    CLOUDSAT_pt1_idx = d1_min[0][0]
    MODIS_pt1_idx = d1_min[0][1]
    
    #----- find end pixel -----#

    modis_pt2 = np.stack((swath_latitudes[modis_dim_along_track,:],swath_longitudes[modis_dim_along_track,:]), axis=-1)

    cloudsat_pt2 = np.stack((cs_latitudes[:,0],cs_longitudes[:,0]), axis=-1)

    d2 = cdist(cloudsat_pt2,modis_pt2)

    d2_min = np.argwhere(d2==d2.min())

    CLOUDSAT_pt2_idx = d2_min[0][0]
    MODIS_pt2_idx = d2_min[0][1]
    
     #----- find all pixels -----#
    MODIS_min_idx = min(MODIS_pt1_idx,MODIS_pt2_idx) - 2
    MODIS_max_idx = max(MODIS_pt1_idx,MODIS_pt2_idx) + 2
    
    lat = swath_latitudes[:,MODIS_min_idx:MODIS_max_idx]
    lat = lat.ravel()
    long = swath_longitudes[:,MODIS_min_idx:MODIS_max_idx]
    long = long.ravel()
    
    modis_pt = np.stack((lat,long), axis=-1)

    cloudsat_pt = np.stack((cs_latitudes[CLOUDSAT_pt1_idx:CLOUDSAT_pt2_idx,0],
                          cs_longitudes[CLOUDSAT_pt1_idx:CLOUDSAT_pt2_idx,0]), axis=-1)

    d = cdist(cloudsat_pt, modis_pt)
    
    res = np.argmin(d, axis=1)

    res = np.unravel_index(res,(modis_granule_shape[0], (MODIS_max_idx-MODIS_min_idx)))
    
    cs_range = (MODIS_min_idx, MODIS_max_idx)
    
    cloudsat_idx = [i for i in range(CLOUDSAT_pt1_idx,CLOUDSAT_pt2_idx)] 
    
    modis_colocated_idx_dim_0 = res[0]
    modis_colocated_idx_dim_1 = res[1]

    modis_colocated_idx_dim_1 = modis_colocated_idx_dim_1 + MODIS_min_idx
    
    data = np.array([modis_colocated_idx_dim_0,modis_colocated_idx_dim_1,cloudsat_idx])
    
    colocated_pixel = np.logical_and.reduce([modis_colocated_idx_dim_0,modis_colocated_idx_dim_1,cloudsat_idx])
    colocated_pixel_idx = np.where(colocated_pixel == 1)
    data = data.transpose()
    
    return cs_range, data, modis_colocated_idx_dim_0, modis_colocated_idx_dim_1, cloudsat_idx

def find_track_range(cs_latitudes, cs_longitudes, latitudes, longitudes):

    i = MAX_HEIGHT // 2

    i_lat, i_lon = latitudes[i-1:i+1, :], longitudes[i-1:i+1, :]
    
    print(i_lat, i_lon)
    
    i_indices = get_track_oi(cs_latitudes, cs_longitudes, i_lat, i_lon)
    
    i_mapping = scalable_align(cs_latitudes[i_indices], cs_longitudes[i_indices], i_lat, i_lon)

    min_j, max_j = min(i_mapping[1]), max(i_mapping[1])
    
    cs_range = (max(0, min_j), min(max_j, MAX_WIDTH - 1))
    
    return cs_range


def map_labels(mapping, labels, shape):

    labelmask = np.zeros((*shape, labels.shape[1]))

    for i, l in enumerate(labels):
        labelmask[mapping[0][i], mapping[1][i]] += l

    return labelmask


#if __name__ == "__main__":

    #example
    #test_lat = np.array([[8., 10., 12.],
                     #[8.1, 10., 12.2],
                     #[8.6, 10.9, 12.1],
                     #[9.6, 11.1, 13.1],
                     #[10.6, 11.9, 13.5]])

    #test_lon = np.array([[10.,  20.,  30.], 
                         #[10.1, 21.1, 33.3], 
                         #[12.9, 22.9, 34.4], 
                         #[14.2, 26.1, 35.5], 
                         #[15.4, 28.9, 36.6]])

    #test_track = np.array([[8.7, 9.1, 10.1, 13.7], [11.1, 18.4, 39.1, 45.9], [1, 3, 7, 6]])

    #mapping = scalable_align(test_track[0], test_track[1], test_lat, test_lon)
    #labels = map_labels(mapping, np.array(test_track[2])[:, None], (5, 3))

    #print(labels)
