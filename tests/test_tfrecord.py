import unittest
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from .utils.tfrecorder import TrainingRecorder
from glob import glob

class TestTFRecorder(unittest.TestCase):

    n, w, h, c  = 1, 128, 128, 4
    n_class     = 5

    def test_x_shape(self):
        records = glob('./saved_dataset/**/**/*.tfrecord')
        recorder = TrainingRecorder(x_shape = [self.w, self.h, self.c], y_shape = [self.w, self.h])
        training_dataset = recorder.create_data_iterator(records, self.n, repeat=False)
        x, y = next(iter(training_dataset))
        assert x.shape == [self.n, self.w, self.h, self.c]
    
    def test_y_shape(self):
        records = glob('./saved_dataset/**/**/*.tfrecord')
        recorder = TrainingRecorder(x_shape = [128, 128, 4], y_shape = [128, 128])
        training_dataset = recorder.create_data_iterator(records, 1, repeat=False)
        x, y = next(iter(training_dataset))
        assert y.shape == [self.n, self.w, self.h, self.n_class]

if __name__ == '__main__':
    unittest.main()
