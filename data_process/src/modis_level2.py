import glob
import numpy as np
import os

from pyhdf.SD import SD, SDC
from satpy import Scene

'''take in the MODIS level 1 filename to get the information needed to find the corresponding MODIS level 2 filename.
This info includes the YYYY and day in year (ex: AYYYYDIY) and then the time of the pass (ex1855)
It returns the full level 2 filename path'''

MAX_WIDTH, MAX_HEIGHT = 1354, 2040

def bits_stripping(bit_start,bit_count,value):
    """Extract specified bit from bit representation of integer value.
    Parameters
    ----------
    bit_start : int
        Starting index of the bits to extract (first bit has index 0)
    bit_count : int
        Number of bits starting from bit_start to extract
    value : int
        Number from which to extract the bits
    Returns
    -------
        int
        Value of the extracted bits
    """
    bitmask=pow(2,bit_start+bit_count)-1
    return np.right_shift(np.bitwise_and(value,bitmask),bit_start)

def get_cloud_mask(l1_filename, cloud_mask_dir):
    
    """ return a mask, with 0 for cloudy, 1 for uncertain/probably cloudy, 2 for probably clear, and 3 for clear. """
    
    basename = os.path.split(l1_filename)

    cloud_mask_filename = glob.glob(os.path.join(cloud_mask_dir, 'MYD35*' + l1_filename.split('.A')[1][:12] + '*'))[0]
    
    # satpy returns(0=Cloudy, 1=Uncertain, 2=Probably Clear, 3=Confident Clear)
    swath = Scene(reader = 'modis_l2', filenames = [cloud_mask_filename])
    swath.load(['cloud_mask'], resolution = 1000)
    
    cloud_mask = np.array(swath['cloud_mask'].load())[:MAX_HEIGHT, :MAX_WIDTH]
    

    cloud_mask = cloud_mask.astype(np.intp)
    
    file = SD(cloud_mask_filename, SDC.READ)
    cloud_flag = file.select('Cloud_Mask')
    
    # There are 6 bytes in the cloudmask:

#     "                                                                          \n",
#     " Bit fields within each byte are numbered from the left:                  \n",
#     " 7, 6, 5, 4, 3, 2, 1, 0.                                                  \n",
#     " The left-most bit (bit 7) is the most significant bit.                   \n",
#     " The right-most bit (bit 0) is the least significant bit.                 \n",
#     "                                                                          \n",
    maskVals1=cloud_flag[0,:,].astype(np.uint8)   #get the first byte
    maskVals2=cloud_flag[1,:,].astype(np.uint8)   #get the second byte
    maskVals3=cloud_flag[2,:,].astype(np.uint8)   #get the third byte
    maskVals4=cloud_flag[3,:,].astype(np.uint8)   #get the fourth byte
    
#     " bit field       Description                             Key              \n",
#     " ---------       -----------                             ---              \n",
#     "                                                                          \n",
#     " 0               Cloud Mask Flag                      0 = Not determined  \n",
#     "                                                      1 = Determined      \n",
#     "                                                                          \n",
#     " 2, 1            Unobstructed FOV Quality Flag        00 = Cloudy         \n",
#     "                                                      01 = Uncertain      \n",
#     "                                                      10 = Probably Clear \n",
#     "                                                      11 = Confident Clear\n",
#     "                 PROCESSING PATH                                          \n",
#     "                 ---------------                                          \n",
#     " 3               Day or Night Path                    0 = Night / 1 = Day \n",
#     " 4               Sunglint Path                        0 = Yes   / 1 = No  \n",
#     " 5               Snow/Ice Background Path             0 = Yes   / 1 = No  \n",
#     " 7, 6            Land or Water Path                   00 = Water          \n",
#     "                                                      01 = Coastal        \n",
#     "                                                      10 = Desert         \n",
#     "                                                      11 = Land           \n",
#     " ____ END BYTE 1 _________________________________________________________\n",    
    
    Bit4 = bits_stripping(3,1,maskVals1)
    Bit5 = bits_stripping(2,1,maskVals1)
    Bit6_7 = bits_stripping(1,2,maskVals1)
#     "                                                                          \n",
#     " bit field       Description                             Key              \n",
#     " ---------       -----------                             ---              \n",
#     "                                                                          \n",
#     "                 ADDITIONAL INFORMATION                                   \n",
#     "                 ----------------------                                   \n",
#     " 0               Non-cloud obstruction Flag              0 = Yes / 1 = No \n",
#     " 1               Thin Cirrus Detected  (Solar)           0 = Yes / 1 = No \n",
#     " 2               Shadow Found                            0 = Yes / 1 = No \n",
#     " 3               Thin Cirrus Detected  (Infrared)        0 = Yes / 1 = No \n",
#     " 4               Adjacent Cloud Detected **              0 = Yes / 1 = No \n",
#     "                 ** Implemented Post Launch to                            \n",
#     "                    Indicate cloud found within                           \n",
#     "                    surrounding 1 km pixels *                             \n",
#     "                                                                          \n",
#     "                 1-km CLOUD FLAGS                                         \n",
#     "                 ----------------                                         \n",
#     " 5               Cloud Flag - IR Threshold               0 = Yes / 1 = No \n",
#     " 6               High Cloud Flag - CO2 Test              0 = Yes / 1 = No \n",
#     " 7               High Cloud Flag - 6.7 micron Test       0 = Yes / 1 = No \n",
#     " ____ END BYTE 2 _________________________________________________________\n",
    Bit8 = bits_stripping(7,1,maskVals2)
    Bit9 = bits_stripping(6,1,maskVals2)
    Bit10 = bits_stripping(5,1,maskVals2)
    Bit11 = bits_stripping(4,1,maskVals2)
    Bit12 = bits_stripping(3,1,maskVals2)
    Bit13 = bits_stripping(2,1,maskVals2)
    Bit14 = bits_stripping(1,1,maskVals2)
    Bit15 = bits_stripping(0,1,maskVals2)
#     "                                                                          \n",
#     " bit field       Description                             Key              \n",
#     " ---------       -----------                             ---              \n",
#     "                                                                          \n",
#     " 0               High Cloud Flag - 1.38 micron Test      0 = Yes / 1 = No \n",
#     " 1               High Cloud Flag - 3.7-12 micron Test    0 = Yes / 1 = No \n",
#     " 2               Cloud Flag - IR Temperature             0 = Yes / 1 = No \n",
#     "                              Difference                                  \n",
#     " 3               Cloud Flag - 3.7-11 micron Test         0 = Yes / 1 = No \n",
#     " 4               Cloud Flag - Visible Reflectance Test   0 = Yes / 1 = No \n",
#     " 5               Cloud Flag - Visible Reflectance        0 = Yes / 1 = No \n",
#     "                              Ratio Test                                  \n",
#     " 6               Cloud Flag - NDVI Final Confidence      0 = Yes / 1 = No \n",
#     "                              Confirmation Test                           \n",
#     " 7               Cloud Flag - Night 7.3-11 micron Test   0 = Yes / 1 = No \n",
#     " ____ END BYTE 3 _________________________________________________________\n",
    Bit16 = bits_stripping(7,1,maskVals3)
    Bit17 = bits_stripping(6,1,maskVals3)
    Bit18 = bits_stripping(5,1,maskVals3)
    Bit19 = bits_stripping(4,1,maskVals3)
    Bit20 = bits_stripping(3,1,maskVals3)
    Bit21 = bits_stripping(2,1,maskVals3)
    Bit22 = bits_stripping(1,1,maskVals3)
    Bit23 = bits_stripping(0,1,maskVals3)
#     "                                                                          \n",
#     " bit field       Description                             Key              \n",
#     " ---------       -----------                             ---              \n",
#     "                                                                          \n",
#     "                 ADDITIONAL TESTS                                         \n",
#     "                 ----------------                                         \n",
#     " 0               Cloud Flag - Spare                      0 = Yes / 1 = No \n",
#     " 1               Cloud Flag - Spatial Variability        0 = Yes / 1 = No \n",
#     " 2               Final Confidence Confirmation Test      0 = Yes / 1 = No \n",
#     " 3               Cloud Flag - Night Water                0 = Yes / 1 = No \n",
#     "                              Spatial Variability                         \n",
#     " 4               Suspended Dust Flag                     0 = Yes / 1 = No \n",
#     "                                                                          \n",
#     " 5-7             Spares                                                   \n",
#     " ____ END BYTE 4 _________________________________________________________\n",
    Bit24 = bits_stripping(7,1,maskVals4)
    Bit25 = bits_stripping(6,1,maskVals4)
    Bit26 = bits_stripping(5,1,maskVals4)
    Bit27 = bits_stripping(4,1,maskVals4)
    Bit28 = bits_stripping(3,1,maskVals4)
    
    mask =[cloud_mask, Bit4, Bit5, Bit6_7, Bit8, Bit9, Bit10, Bit11, Bit12, Bit13, Bit14, Bit15, Bit16, Bit17, Bit18, Bit19, Bit20, Bit21, Bit22, Bit23, Bit24, Bit25, Bit26, Bit27, Bit28]
    
    
    return mask



if __name__ == "__main__":
    
    l1_path = sys.argv[1]
    cloudmask_dir = "/Users/documents/Desk/Antarctic_sea_ice/Dataset_test/MODIS/MYD35_L2/"

    save_dir = "/Users/documents/Desk/Antarctic_sea_ice/Dataset_test/MODIS/MYD35_L2/data-processed"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cloudmask = get_cloud_mask(l1_path, cloudmask_dir)

    np.save(os.path.join(save_dir, os.path.basename(l1_path).replace(".hdf", ".npy")), cloudmask)
