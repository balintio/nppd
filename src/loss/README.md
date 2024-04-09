Perceptual loss
===============

This folder contains our implementation of the perceptual loss presented in [[Thomas2022]](https://www.intel.com/content/www/us/en/developer/articles/technical/temporally-stable-denoising-and-supersampling.html).

How to use
----------
Import the `Features` class from `loss.py` to use the loss, after instantiating you can call `spatial_loss` and `temporal_loss`. See `model.py` for an example of how NPPD uses these.

How to train
----------

1. Download color and semantic maps from ml-hypersim using [Thomas Germer's script](https://github.com/apple/ml-hypersim/tree/main/contrib/99991)

2. Run `process_hypersim.py` to transform the training data to Zarr format

3. Now you can train the perceptual loss running `python loss.py`

You can skip the first two steps by downloading our [Zarr file](https://neural-partitioning-pyramids.mpi-inf.mpg.de/data/hypersim_zarr.zip).

Note that `loss.py` doesn't import the other Python files for inference; you only need those for training.