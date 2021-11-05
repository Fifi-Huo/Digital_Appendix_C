import netCDF4 as nc4
import numpy as np

radiances = ['ev_250_aggr1km_refsb_1', 'ev_250_aggr1km_refsb_2', 'ev_500_aggr1km_refsb_3', 'ev_500_aggr1km_refsb_4', 'ev_500_aggr1km_refsb_5', 'ev_500_aggr1km_refsb_6', 'ev_500_aggr1km_refsb_7', 'ev_1km_refsb_8', 'ev_1km_refsb_9', 'ev_1km_refsb_10', 'ev_1km_refsb_11', 'ev_1km_refsb_12', 'ev_1km_refsb_13L', 'ev_1km_refsb_13H', 'ev_1km_refsb_14L', 'ev_1km_refsb_14H', 'ev_1km_refsb_15', 'ev_1km_refsb_16', 'ev_1km_refsb_17', 'ev_1km_refsb_18', 'ev_1km_refsb_19',   'ev_1km_emissive_20', 'ev_1km_emissive_21', 'ev_1km_emissive_22', 'ev_1km_emissive_23', 'ev_1km_emissive_24', 'ev_1km_emissive_25', 'ev_1km_refsb_26', 'ev_1km_emissive_27', 'ev_1km_emissive_28', 'ev_1km_emissive_29', 'ev_1km_emissive_30', 'ev_1km_emissive_31', 'ev_1km_emissive_32', 'ev_1km_emissive_33', 'ev_1km_emissive_34', 'ev_1km_emissive_35', 'ev_1km_emissive_36']
coordinates = ['latitude', 'longitude']
#properties = ['cloud_water_path', 'cloud_optical_thickness', 'cloud_effective_radius', 'cloud_phase_optical_properties', 'cloud_top_pressure', 'cloud_top_height', 'cloud_top_temperature', 'cloud_emissivity', 'surface_temperature']
rois = 'cloud_mask'
labels = 'cloud_occurrences'

def read_nc(nc_file):
    """return masked arrays, with masks indicating the invalid values"""
    
    file = nc4.Dataset(nc_file, 'r', format='NETCDF4')

    f_radiances = np.vstack([file.variables[name][:] for name in radiances])
    #f_properties = np.vstack([file.variables[name][:] for name in properties])
    f_rois = file.variables[rois][:]
    f_labels = file.variables[labels][:]

    return f_rois, f_labels