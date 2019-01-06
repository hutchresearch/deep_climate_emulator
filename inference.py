import argparse
import netCDF4 as nc
import numpy as np
import emulator.data as data
import emulator.forecast as forecast
from emulator.normalization import Normalizer

# Number of days in each month for 'noleap' or '365_day' calendar attribute.
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


def find_midpoint(low, high):
    """
    Find the midpoint between two numbers. Expects low <= high.

    Args:
        low (int): Low number
        high (int): High number

    Returns:
        (int): midpoint between low and high
    """
    if high < low:
        raise ValueError("Expected arg 'low' to be less than arg 'high'")

    return low + (high - low) / 2


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
        u_bound = l_bound+noleap_days[month]
        bnds.append([l_bound, u_bound])
        l_bound = u_bound
        month = (month % 12) + 1

    return np.asarray(bnds)


def export_netcdf(pr_field, pr_units, times, time_bnds, time_units,
                  lats, lat_bnds, lons, lon_bnds, filename):
    """
    Export forecast data to NetCDF file.
    NOTE: This function has some hard-coded metadata specific to the CanESM2
    1pctCO2 climate simulation.


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


def build_input_tensor(pr_data, window_size, num_preforecasts, model=None):
    """
    Constructs the first input tensor used for forecasting. Takes
    precipitation maps from the end of the array by default.

    Args:
        pr_data (ndarray): Array of chronologically ordered precipitation maps
        window_size (int): Size of the input window
        num_preforecasts (int): Number of preforecasts
        model (str): Path to model file(s)

    Returns:
        (ndarray) Input tensor for forecast generation.
    """

    if num_preforecasts is not None:

        assert len(pr_data) >= (window_size + num_preforecasts), \
            "Insufficient data for specified window and preforecast lengths."

        print("Generating preforecasts...")
        input_tensor = pr_data[-(window_size + num_preforecasts):-num_preforecasts]
        preforecasts = forecast.gen_forecasts(input_tensor=input_tensor,
                                              num_forecasts=num_preforecasts,
                                              model_path=model)
        print("Done.")

        # Incorporate preforecasts in the first input tensor
        input_tensor = np.concatenate((pr_data[-window_size:-num_preforecasts],
                                       preforecasts))

    else:
        assert len(pr_data) >= window_size, \
            "Insufficient data for specified window length."

        input_tensor = pr_data[-window_size:]

    return input_tensor


def main(args):
    print("Loading precipitation data...")

    # Load in precipitation maps and other metadata from NetCDF file.
    nc_file = nc.Dataset(args.data)
    pr_data = nc_file.variables['pr'][:]

    # Normalize precipitation data
    norm = Normalizer()
    pr_data_normalized = norm.transform(pr_data, len(pr_data))

    # Build input tensor (generating preforecasts if necessary)
    input_tensor = build_input_tensor(pr_data_normalized, args.window_size,
                                      args.num_preforecasts, args.model)

    # Generate Forecasts
    print('Generating precipitation forecasts...')
    forecasts = forecast.gen_forecasts(input_tensor=input_tensor,
                                       num_forecasts=args.num_forecasts,
                                       model_path=args.model)
    print("Done.")

    # Denormalize forecasts
    forecasts = norm.inverse_transform(forecasts)

    # Create time coordinate
    last_date = nc.num2date(nc_file.variables['time'][:],
                            nc_file.variables['time'].units)[-1]
    time_units = 'days since %d-%d-%d' % \
                 (last_date.year, last_date.month+1, 1)
    time_bnds = gen_month_time_bnds(start_month=last_date.month + 1,
                                    num_forecasts=args.num_forecasts)
    times = [find_midpoint(x[0], x[1]) for x in time_bnds]

    # Export precipitation maps, reusing some metadata from the input file.
    if args.mm_per_day:
        forecasts = data.convert_to_mm_per_day(forecasts)
        export_netcdf(pr_field=forecasts,
                      pr_units='mm day-1',
                      times=times,
                      time_bnds=time_bnds,
                      time_units=time_units,
                      lats=nc_file.variables['lat'],
                      lat_bnds=nc_file.variables['lat_bnds'],
                      lons=nc_file.variables['lon'],
                      lon_bnds=nc_file.variables['lon_bnds'],
                      filename=args.outfile)
    else:
        export_netcdf(pr_field=forecasts,
                      pr_units='kg m-2 s-1',
                      times=times,
                      time_bnds=time_bnds,
                      time_units=time_units,
                      lats=nc_file.variables['lat'],
                      lat_bnds=nc_file.variables['lat_bnds'],
                      lons=nc_file.variables['lon'],
                      lon_bnds=nc_file.variables['lon_bnds'],
                      filename=args.outfile)

    nc_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data",
                        type=str,
                        required=True,
                        help="NetCDF file containing precipitation data.")
    parser.add_argument("--model",
                        type=str,
                        help="Directory containing TensorFlow checkpoint "
                             "files.")
    parser.add_argument("--window_size",
                        type=int,
                        help="Specifies the number of precipitation maps "
                             "the model requires for each input.",
                        default=60)
    parser.add_argument("--num_forecasts",
                        type=int,
                        help="Specifies the number of forecasts (i.e. the "
                             "number future time steps you wish to generate "
                             "forecasts for).",
                        default=120)
    parser.add_argument("--num_preforecasts",
                        type=int,
                        help="Specifies the number of preforecasts (i.e. "
                             "predictions preceding the first forecast date). "
                             "This can alleviate high error in early "
                             "forecasts for models trained with scheduled "
                             "sampling.")
    parser.add_argument("--mm_per_day",
                        action='store_true',
                        default=False,
                        help="Convert precipitation units to mm day^-1 ("
                             "assumes precipitation data is in kg m^-2 s^-1)")
    parser.add_argument("--outfile",
                        type=str,
                        default='forecasts.nc',
                        help="Name of output NetCDF file containing "
                             "precipitation forecasts.")

    main(parser.parse_args())
