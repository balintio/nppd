from multiprocessing.pool import Pool
import re
import os
import itertools
import zarr
import numpy as np
import h5py
import os

zarr.blosc.use_threads = False

def getSequences(path, folders, files):
    m = re.match(r'\.\/hypersim\/ai_(\d{3}_\d{3})\/images\/scene_cam_(\d{2})_geometry_hdf5', path)
    if m and len(files) == 100:
        return [(m[1], m[2])]
    else:
        return []

seq = list(itertools.chain(*[getSequences(*f) for f in os.walk('./hypersim')]))

def compressRGBE(color):
    log_radiance = np.log(color[np.where(color > 0)])

    if log_radiance.size == 0: # Handle black frames
        return np.zeros((4, color.shape[1], color.shape[2]), dtype=np.uint8), [0, 0]
    
    # Calculate exposure
    min_exp = np.min(log_radiance)
    max_exp = np.max(log_radiance)

    # Get exponent from brightest channel
    brightest_channel = np.max(color, axis = 0)
    exponent = np.ones_like(brightest_channel) * -np.inf
    np.log(brightest_channel, out=exponent, where=brightest_channel > 0)

    # Quantise exponent with ceiling function
    e_channel = np.minimum((exponent - min_exp) / (max_exp - min_exp) * 256, 255).astype(np.uint8)[np.newaxis]
    # Actually encoded exponent
    exponent = np.exp(((e_channel.astype(np.float32) + 1)/256) * (max_exp - min_exp) + min_exp)

    # Quantise colour channels
    rgb_float = (color / exponent) * 255
    rgb_channels = (rgb_float).astype(np.uint8)
    # Add dither (exponents were quantised with ceiling so this doesn't go over 255)
    rgb_channels += ((rgb_float - rgb_channels) > 0.5)

    return np.concatenate([rgb_channels, e_channel]), [min_exp, max_exp]

def saveSequence(id):
    zarr_file = f'./hypersim_zarr/ai_{id[0]}_{id[1]}.zip'
    if os.path.exists(zarr_file):
        return

    semantic = np.zeros((100, 768, 1024), dtype=np.int32)
    color = np.zeros((100, 768, 1024, 3), dtype=np.float32)

    try:
        for frame in range(100):
            with h5py.File(f'./hypersim/ai_{id[0]}/images/scene_cam_{id[1]}_geometry_hdf5/frame.{frame:04d}.semantic.hdf5') as f:
                semantic[frame] = f['dataset']
            with h5py.File(f'./hypersim/ai_{id[0]}/images/scene_cam_{id[1]}_final_hdf5/frame.{frame:04d}.color.hdf5') as f:
                color[frame] = f['dataset']
    except OSError:
        print(f'Issue with sequence ai_{id[0]}_{id[1]}')
        return
    
    with zarr.ZipStore(zarr_file, mode='w') as store:
        f = zarr.group(store=store)
        compressor = zarr.Blosc(cname='lz4hc', clevel=9, shuffle=2)

        color = np.transpose(color, (0, 3, 1, 2))
        color = np.where(np.logical_and(np.isfinite(color), color > 0), color, 0)
        rgbe, exposure = zip(*map(compressRGBE, color))

        f.array('semantic', data=semantic, chunks=(1, 128, 128), compressor=compressor)
        f.array('color', data=rgbe, chunks=(1, 4, 128, 128), compressor=compressor)
        f.array('exposure', data=exposure)

#saveSequence(seq[0])
with Pool(64) as p:
    p.map(saveSequence, seq)

# Issue with sequence ai_007_005_01
# Issue with sequence ai_042_001_01


