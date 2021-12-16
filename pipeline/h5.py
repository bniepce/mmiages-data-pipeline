import numpy as np
import h5py

class HDF5Store(object):
    def __init__(self, datapath, datasets, shapes, dtype, compression="gzip", chunk_len=1):
        self.datapath = datapath
        self.datasets = datasets
        self.shapes = shapes
        self.i = {datasets[0]: 0, datasets[1]: 0}
        
        with h5py.File(self.datapath, mode='w') as h5f:
            for idx, i in enumerate(datasets):
                self.dset = h5f.create_dataset(
                    i,
                    shape=(0, ) + shapes[idx],
                    maxshape=(None, ) + shapes[idx],
                    dtype=dtype[idx],
                    compression=compression,
                    chunks=(chunk_len, ) + shapes[idx])
    
    def append(self, dataset, values, shape):
        with h5py.File(self.datapath, mode='a') as h5f:
            dset = h5f[dataset]
            dset.resize((self.i[dataset] + 1, ) + shape)
            dset[self.i[dataset]] = [values]
            self.i[dataset] += 1
            h5f.flush()