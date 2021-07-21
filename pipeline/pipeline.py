import os
from .processing import get_scans, convert_scans
from glob import glob

class BraTSPipeline(object):
    
    def __init__(self, data_path):
        
        self._data_path = data_path
        self._operations = [get_scans, convert_scans]
    
    def add_operation(self, ops : callable):
        if callable(ops):
            self._operations.append(ops)
    
    def process(self, mode : str):
        source = glob(os.path.join(self._data_path, '{}/*'.format(mode)))
        for ops in self._operations:
            source = ops(source)
        return source
        
