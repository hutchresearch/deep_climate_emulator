# deep_climate_emulator

A Deep Neural Network approach for estimating precipitation fields in Earth
System Models.

## To install package
```pip install git+https://github.com/webert3/climate-emulator```

## Training models
See ```trainer.py``` for an example.

## Using pretrained models to generate precipitation forecasts
See ```inference.py``` for an example. 

This package comes bundled with a pretrained 18-layer Residual Network (window
size = 60), and this will be loaded by default if TensorFlow checkpoint files
are not provided.

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
