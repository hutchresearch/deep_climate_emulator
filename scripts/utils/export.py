import netCDF4 as nc

# Number of days in each month. Used for '365_day' calendar attribute.
import numpy as np

noleap_days = {
    1: 31,
    2: 28,
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31
}


def to_netcdf(pr_field, pr_units, times, time_bnds, time_units,
              lats, lat_bnds, lons, lon_bnds, filename):
    """
    Export forecast data to NetCDF file.
    NOTE: This function uses metadata specific to the CanESM2 1pctCO2 climate
    simulation.


    Args:
        pr_field (ndarray): Array of precipitation forecasts (chronological)
        pr_units (str): Precipitation units
        times (ndarray): Array of ints representing the number of days since
            the start date defined in 'time units'
        time_bnds (ndarray): Boundaries corresponding to values in 'time' array
        time_units (str): Units relative to start date (following Unidata
            conventions)
        lats (netCDF4.Variable): Latitude data
        lat_bnds (netCDF4.Variable): Boundaries for latitude data
        lons (netCDF4.Variable): Longitude data
        lon_bnds (netCDF4.Variable): Boundaries for longitude data
        filename (str): Name of output NetCDF file
    """
    if len(pr_field) != len(times):
        raise ValueError("Mismatch in length of precipitation field and "
                         "datetime collections. Should be equal.")

    rootgrp = nc.Dataset("%s" % filename, 'w', format='NETCDF4')

    # Dimensions
    rootgrp.createDimension('lat', 64)
    rootgrp.createDimension('lon', 128)
    rootgrp.createDimension('bnds', 2)
    rootgrp.createDimension('time', None)

    # Variables
    lat_out = rootgrp.createVariable('lat', 'f8', ('lat',))
    lat_bnds_out = rootgrp.createVariable('lat_bnds', 'f8', ('lat', 'bnds',))
    lon_out = rootgrp.createVariable('lon', 'f8', ('lon',))
    lon_bnds_out = rootgrp.createVariable('lon_bnds', 'f8', ('lon', 'bnds',))
    time_out = rootgrp.createVariable('time', 'f8', ('time',))
    time_bnds_out = rootgrp.createVariable('time_bnds', 'f8', ('time', 'bnds',))
    pr_out = rootgrp.createVariable('pr', 'f8', ('time', 'lat', 'lon'))

    # Attributes
    rootgrp.description = "Precipitation forecasts emulating idealized " \
                          "precipitation output from the CanESM2 Earth System " \
                          "Model CanESM2."
    rootgrp.source = "deep_climate_emulator python package."

    lat_out.units = "degrees_north"
    lat_out.axis = "Y"
    lat_out.long_name = "latitude"
    lat_out.standard_name = "latitude"
    lat_out.bounds = 'lat_bnds'

    lon_out.units = "degrees_east"
    lon_out.axis = "X"
    lon_out.long_name = "longitude"
    lon_out.standard_name = "longitude"
    lon_out.bounds = 'lon_bnds'

    time_out.units = time_units
    time_out.calendar = "365_day"
    time_out.axis = "T"
    time_out.long_name = "time"
    time_out.standard_name = "time"
    time_out.bounds = 'time_bnds'

    pr_out.units = pr_units
    pr_out.long_name = "Precipitation"
    pr_out.standard_name = "precipitation_flux"
    pr_out.comment = "at surface; includes both liquid and solid phases from " \
                     "all types of clouds (both large-scale and convective)"

    lat_out[:] = lats[:]
    lon_out[:] = lons[:]
    lat_bnds_out[:] = lat_bnds[:]
    lon_bnds_out[:] = lon_bnds[:]
    time_bnds_out[:] = time_bnds
    time_out[:] = times
    pr_out[:] = pr_field

    rootgrp.close()
    print('Exported data to %s' % filename)


def gen_month_time_bnds(start_month, num_forecasts):
    """
    Create the time bounds for monthly time series.

    Args:
        start_month (int): In range [1, 12], representing the month
        num_forecasts (int): Number of forecasts generated

    Returns:
        (ndarray): Array of time bounds with shape (2, num_forecasts)
    """
    bnds = []
    l_bound = 0
    month = start_month
    for i in range(num_forecasts):
        u_bound = l_bound + noleap_days[month]
        bnds.append([l_bound, u_bound])
        l_bound = u_bound
        month = (month % 12) + 1

    return np.asarray(bnds)