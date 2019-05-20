import argparse
import json
import netCDF4 as nc
from emulator.model import ResNet


def main(args):
    print("Loading precipitation data...")
    nc_file = nc.Dataset(args.data)
    pr_data = nc_file.variables['pr'][:]

    print("Building graph...")
    model = json.loads(open(args.architecture, 'r').read())
    hyperparameters = json.loads(open(args.hyperparameters, 'r').read())
    resnet = ResNet(architecture=model, hyperparameters=hyperparameters,
                    pr_field=pr_data, train_pct=0.7)

    mse_loss = resnet.fit()
    print("Dev Set MSE Loss=%.3f" % mse_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data",
                        type=str,
                        required=True,
                        help="NetCDF file containing precipitation data.")
    parser.add_argument("--architecture",
                        type=str,
                        required=True,
                        help="JSON file defining the DNN architecture.")
    parser.add_argument("--hyperparameters",
                        type=str,
                        required=True,
                        help="JSON file defining the hyperparameter "
                             "configuration.")

    main(parser.parse_args())
