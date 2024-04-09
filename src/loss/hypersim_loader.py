import torch
import os
import numpy as np
import zarr
import lightning as L
from noisebase.compression import decompress_RGBE

from mmap_zipstore import MMapZipStore

WINDOW = 256
WIDTH = 1024
HEIGHT = 768
LENGTH = 100
FOLDER = './hypersim_zarr/'

class HypersimDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

        self.files = list(map(lambda path: FOLDER + path, os.listdir(FOLDER)))
        self.store_function = zarr.ZipStore
    
    def __getitem__(self, idx):
        frame_idx = idx % LENGTH
        sequence_idx = idx // LENGTH

        ds = zarr.group(store = self.store_function(self.files[sequence_idx], mode='r'))

        # TODO: Don't augment the validation set
        rng = np.random.default_rng()
        h = rng.integers(HEIGHT - WINDOW)
        w = rng.integers(WIDTH - WINDOW)

        frame = {
            'semantic': ds['semantic'][frame_idx, h:h+WINDOW, w:w+WINDOW],
            'color': decompress_RGBE(
                ds['color'][frame_idx, :, h:h+WINDOW, w:w+WINDOW],
                ds['exposure'][frame_idx, :]
            )
        }
        
        ds.store.close()

        return frame

    def __len__(self):
        return len(self.files) * LENGTH

# Inject mmap zarr store into worker processes
# Caches dataset in RAM before decompression
# Also needs persistent_workers=True
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    ds = worker_info.dataset.dataset
    ds.store_cache = {}
    ds.store_function = lambda *args, **kwargs: MMapZipStore(ds.store_cache, *args, **kwargs)

class HypersimDataModule(L.LightningDataModule):

    def setup(self, stage):
        ds = HypersimDataset()
        self.train_set, self.val_set = torch.utils.data.random_split(ds, [0.95, 0.05])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_set, 
            batch_size=64,
            shuffle=True,
            num_workers=8,
            worker_init_fn=worker_init_fn, 
            persistent_workers=True,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_set, 
            batch_size=64, 
            num_workers=8,
            worker_init_fn=worker_init_fn, 
            persistent_workers=True, 
            pin_memory=True
        )

    