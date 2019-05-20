# deep_climate_emulator

A Deep Neural Network approach for estimating precipitation fields in Earth
System Models.

## To install package
```pip install git+https://github.com/hutchresearch/deep_climate_emulator```

## Training models
We provide ```trainer.py``` as an example script that utilizes our residual network training 
package. To train a model using the same architecture and hyperparameters used in our 18-Layer 
Residual Network, run the following:

```
# Pull scripts to local machine, if not already available.
git clone https://github.com/hutchresearch/deep_climate_emulator 

# Install the "deep_climate_emulator" package, if not already installed.
pip install git+https://github.com/hutchresearch/deep_climate_emulator

# Navigate to "scripts" directory and run "trainer.py".
cd deep_climate_emulator/scripts
python trainer.py \
--data <PATH_TO_PRECIPITATION_DATA> \
--architecture ../configs/18-layer-ResNet_architecture.json \
--hyperparameters ../configs/18-layer-ResNet_hyperparameters.json
```

_Note:_ Our package was developed using Python 3. Python 2 compatibility is not 
guaranteed.

## Using pretrained models to generate precipitation forecasts
We provide ```inference.py``` as an example script for generating predictions with 
a pretrained model. To generate predictions with our 18-Layer Residual Network run 
the following:

```
# Pull scripts to local machine, if not already available.
git clone https://github.com/hutchresearch/deep_climate_emulator 

# Install the "deep_climate_emulator" package, if not already installed.
pip install git+https://github.com/hutchresearch/deep_climate_emulator

# Navigate to "scripts" directory and run "inference.py".
cd deep_climate_emulator/scripts
python inference.py \
--data <PATH_TO_PRECIPITATION_DATA> \
--window_size 60 \
--num_forecasts 120 \
--num_preforecasts 30 \
--outfile <OUTPUT_FILENAME>
```

This package comes bundled with a pretrained 18-layer Residual Network (window
size = 60), and this will be loaded by default if TensorFlow checkpoint files
are not provided with the `--model` flag.

## GPU support

It is highly recommended to train your models with a GPU! To train using a GPU,
make sure you have the appropriate versions of TensorFlow and cuDNN installed on
your system. For more information on GPU support see:
<https://www.tensorflow.org/install/gpu>

Inference using pretrained models, on the other hand, will run sufficiently fast
on CPUs alone (i.e. without GPU support). Our package uses a version of
TensorFlow that runs on CPUs by default.

## Training Data
All climate model output used in this study as training data is available through 
the Earth System Grid Federation.

## Citations
