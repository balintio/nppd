import os
import zipfile
import zarr
import mmap
from threading import RLock

# Why doesn't mmap have this??
class SeekableMMap(mmap.mmap):
    def seekable(self):
        return True

# Mainly copied from https://github.com/zarr-developers/zarr-python/blob/76ba69a21018822a5a0244c03af882a09293ff28/zarr/storage.py#L1763
class MMapZipStore(zarr.ZipStore):
    def __init__(self, cache, path, compression=zipfile.ZIP_STORED, allowZip64=True, mode='a',
                 dimension_separator=None):
        assert mode == 'r', 'Can only cache read only stores.'

        path = os.path.abspath(path)
        self.path = path
        self.compression = compression
        self.allowZip64 = allowZip64
        self.mode = mode
        self._dimension_separator = dimension_separator
        self.mutex = RLock()

        if path in cache:
            self.zf = cache[path]
        else:
            fd = os.open(path, os.O_RDONLY)
            file = SeekableMMap(fd, 0, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ) # Shared between loader processes
            self.zf = zipfile.ZipFile(file, mode=mode, compression=compression, allowZip64=allowZip64)

            # Cache handles instead of closing and reopening
            # Loads the entire dataset into memory over time

            # TODO: close the handles on worker shutdown

            cache[path] = self.zf
    
    def close(self):
        pass