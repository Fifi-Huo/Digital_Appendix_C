import glob
import numpy as np
import os

from pyhdf.SD import SD, SDC
from satpy import Scene

'''take in the MODIS level 1 filename to get the information needed to find the corresponding MODIS level 2 filename.
This info includes the YYYY and day in year (ex: AYYYYDIY) and then the time of the pass (ex1855)
It returns the full level 2 filename path'''

MAX_WIDTH, MAX_HEIGHT = 1354, 2040


def get_cloud_mask(l1_filename, cloud_mask_dir):
    
    """ return a mask, with 0 for cloudy, 1 for uncertain/probably cloudy, 2 for probably clear, and 3 for clear. """
    
    basename = os.path.split(l1_filename)

    cloud_mask_filename = glob.glob(os.path.join(cloud_mask_dir, 'MYD35*' + l1_filename.split('.A')[1][:12] + '*'))[0]
    
    # satpy returns(0=Cloudy, 1=Uncertain, 2=Probably Clear, 3=Confident Clear)
    swath = Scene(reader = 'modis_l2', filenames = [cloud_mask_filename])
    swath.load(['cloud_mask'], resolution = 1000)
    
    cloud_mask = np.array(swath['cloud_mask'].load())[:MAX_HEIGHT, :MAX_WIDTH]

    cloud_mask = cloud_mask.astype(np.intp)
    
    return cloud_mask



if __name__ == "__main__":
    
    #l1_path = sys.argv[1]
    l1_path = '/Users/documents/Desk/Antarctic_sea_ice/Dataset_test/MODIS/MYD021KM/2016/08/01/MYD021KM.A2016214.1635.061.2018060134603.hdf'
    cloudmask_dir = "/Users/documents/Desk/Antarctic_sea_ice/Dataset_test/MODIS/MYD35_L2/"

    save_dir = "/Users/documents/Desk/Antarctic_sea_ice/Dataset_test/MODIS/MYD35_L2/data-processed"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cloudmask = get_cloud_mask(l1_path, cloudmask_dir)

    np.save(os.path.join(save_dir, os.path.basename(l1_path).replace(".hdf", ".npy")), cloudmask)
