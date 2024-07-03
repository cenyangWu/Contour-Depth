"""Weather forecast ensembles.
Data is the same used in the CVP paper.
"""
from time import time
from pathlib import Path
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from skimage.measure import find_contours
import skimage.io

def load_data(data_dir, config_id=0, verbose=False):
    """ Loads data.
    """
    data_dir = Path(data_dir)
    assert data_dir.exists()

    # Setting variables
    # Fig 1: 20121015; height: 500 hPa; 5600 m contour
    # Fig 2: 20121017; height: 925 hPa; 680 m contour

    time, plev, height, isoval = [
        ["20121015", "120", 500, 5600],  # time, pressure level, height, contour
        ["20121017", "108", 925, 680]
    ][config_id]

    # Loading and preprocessing data

    f = data_dir.joinpath(f"{time}_00_ecmwf_ensemble_forecast.PRESSURE_LEVELS.EUR_LL10.{plev}.pl.nc")
    assert f.exists()

    rootgrp = Dataset(f, "r", "NETCDF4")

    if verbose:
        print("Data model: ")
        print(rootgrp.data_model)
        print()
        print("Dimensions: ")
        for dimobj in rootgrp.dimensions.values():
            print(dimobj)
        # print()
        # print("Variables: ")
        # for varobj in rootgrp.variables.values():
        #     print(varobj)
        print()

    geopot = rootgrp["Geopotential_isobaric"][...]
    geopot = geopot / 9.81
    geopot = geopot.squeeze()

    lat = rootgrp["lat"][...]
    lon = rootgrp["lon"][...]
    isobaric = rootgrp["isobaric"][...]

    if verbose:
        print(rootgrp["Geopotential_isobaric"])
        print()
        print(lat.shape, lat[0], lat[-1])  # latitude is y-axis/rows
        print(lon.shape, lon[0], lon[-1])  # longitude is x-axis/cols
        print()

    ##########################
    # Full ensemble analysis #
    ##########################

    height_level = np.where(isobaric == height)[0]  # slice height we are interested in

    geopot = geopot[:, height_level, :, :].squeeze()
    geopot = np.moveaxis(geopot, [0, 1, 2], [0, 1, 2])

    geopot = np.flip(geopot, axis=1)  # flip x axis
    lat = np.flip(lat)  # flip x axis

    bin_masks = []
    for gp in geopot:
        bin_masks.append(np.zeros_like(gp))
        bin_masks[-1][gp <= isoval] = 1  # we extract the iso line

    rootgrp.close()

    return bin_masks