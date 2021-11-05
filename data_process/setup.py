import os, shutil
from setuptools import setup, find_packages

#
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(CUR_PATH, 'build')
if os.path.isdir(path):
    print('INFO del dir ', path) 
    shutil.rmtree(path)


setup(
    name = 'pipeline',
    # Author details
    author='JiejunHuo',
    author_email='Jiejunh@utas.edu.au',
    version = '0.1', 
    description='Creating MODIS and 2B-CLDCLASS-lidar co-located files (following the earlier work by Zantedeschi et al. (2019))',
    packages = find_packages('src','netcdf'), 
    package_data = {
        # include the *.nc in the netcdf folder
        'netcdf': ['*.nc'],
    },

    include_package_data = True, 
    #exclude_package_data = {'docs':['1.txt']},
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: System :: Logging',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    py_modules=["pipeline"],
    install_requires = [
        'netCDF4==1.5.1.2',
        'scikit-learn==0.20.0',
        'scipy==1.1.0',
    ],

)