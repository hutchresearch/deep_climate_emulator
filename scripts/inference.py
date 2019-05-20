import argparse

import netCDF4 as nc
import numpy as np

import emulator.data as data
import emulator.forecast as forecast
import utils.export as export

from emulator.normalization import Normalizer


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
        raise ValueError("Expected arg \"low\" to be less than arg \"high\"")

    return low + (high - low) / 2


def build_input_tensor(pr_data, window_size, num_preforecasts, model=None):
    """
    Constructs the first input tensor used for forecasting. Takes
    precipitation maps from the end of the array by default (assuming data is
    already sorted in time order).

    Args:
        pr_data (ndarray): Array of chronologically ordered precipitation maps
        window_size (int): Size of the input window
        num_preforecasts (int): Number of preforecasts
        model (str): Path to model file(s)

    Returns:
        (ndarray) Input tensor for forecast generation.
    """

    if num_preforecasts is not None and num_preforecasts > 0:

        assert len(pr_data) >= (window_size + num_preforecasts), \
            "Insufficient data for specified window and preforecast lengths."

        print("Generating preforecasts...")
        input_tensor = pr_data[
                       -(window_size + num_preforecasts):-num_preforecasts]
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
    pr_data = nc_file.variables["pr"][:]

    # Normalize precipitation data
    norm = Normalizer()
    pr_data_normalized = norm.transform(data=pr_data,
                                        train_len=len(pr_data))

    # Build input tensor (generating preforecasts if necessary)
    input_tensor = build_input_tensor(pr_data_normalized, args.window_size,
                                      args.num_preforecasts, args.model)

    # Generate Forecasts
    print("Generating precipitation forecasts...")
    forecasts = forecast.gen_forecasts(input_tensor=input_tensor,
                                       num_forecasts=args.num_forecasts,
                                       model_path=args.model)
    print("Done.")

    # Denormalize forecasts
    forecasts = norm.inverse_transform(forecasts)

    # Create time coordinate
    last_date = nc.num2date(nc_file.variables["time"][:],
                            nc_file.variables["time"].units)[-1]
    time_units = "days since %d-%d-%d" % \
                 (last_date.year, last_date.month + 1, 1)
    time_bnds = export.gen_month_time_bnds(start_month=last_date.month + 1,
                                           num_forecasts=args.num_forecasts)
    times = [find_midpoint(x[0], x[1]) for x in time_bnds]

    # Export precipitation maps, reusing some metadata from the input file.
    if args.mm_per_day:
        forecasts = data.convert_to_mm_per_day(forecasts)
        export.to_netcdf(pr_field=forecasts,
                         pr_units="mm day-1",
                         times=times,
                         time_bnds=time_bnds,
                         time_units=time_units,
                         lats=nc_file.variables["lat"],
                         lat_bnds=nc_file.variables["lat_bnds"],
                         lons=nc_file.variables["lon"],
                         lon_bnds=nc_file.variables["lon_bnds"],
                         filename=args.outfile)
    else:
        export.to_netcdf(pr_field=forecasts,
                         pr_units="kg m-2 s-1",
                         times=times,
                         time_bnds=time_bnds,
                         time_units=time_units,
                         lats=nc_file.variables["lat"],
                         lat_bnds=nc_file.variables["lat_bnds"],
                         lons=nc_file.variables["lon"],
                         lon_bnds=nc_file.variables["lon_bnds"],
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
                        action="store_true",
                        default=False,
                        help="Convert precipitation units to mm day^-1 ("
                             "assumes precipitation data is in kg m^-2 s^-1)")
    parser.add_argument("--outfile",
                        type=str,
                        default="forecasts.nc",
                        help="Name of output NetCDF file containing "
                             "precipitation forecasts.")

    main(parser.parse_args())
