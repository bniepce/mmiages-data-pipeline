import tensorflow as tf
from abc import ABC, abstractmethod
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class AbstractRecorder(ABC):
    """Class to read tfrecord and create tensorflow datasets
    ...
    Attributes
    ----------
    data_shape : tuplue 
        Shape of data registered in tfrecord files
    x_dtype : str
        Type to use for data array
    y_dtype: str
        Type to use for label array
    """
    def __init__(self, data_shape):
        super().__init__()
        self.shape = data_shape
        self.x_dtype = tf.float32
        self.y_dtype = tf.uint8
    
    @abstractmethod
    def parse_function(self):
        raise NotImplementedError
    
    @abstractmethod
    def create_data_iterator(self):
        raise NotImplementedError


class TrainingRecorder(AbstractRecorder):
    """Class to read tfrecords and create the training dataset
    ...
    Attributes
    ----------
    data_shape : tuple
        Shape of the input data
    x_dtype : str
        Type to use for data array
    y_dtype: str
        Type to use for label array
    """
    def __init__(self, data_shape):
        super().__init__(data_shape)

    def parse_function(self, proto):
        """Parse function to create the training Dataset object
        
        Parameters
        ----------
        proto : str
            A scalar string Tensor, single serialized tf Example.
        """
        features = {
            'image': tf.io.FixedLenFeature([], dtype=tf.string),
            'ground_truth': tf.io.FixedLenFeature([128, 128], dtype=tf.int64)
        }
        record = tf.io.parse_single_example(proto, features)
        image_raw = tf.io.decode_raw(record['image'], self.x_dtype)
        image_raw = tf.reshape(image_raw, shape=[4, 128, 128])

        label_raw = tf.one_hot(record["ground_truth"], depth = 5)
        label_raw = tf.cast(record["ground_truth"], self.y_dtype)

        return image_raw, label_raw

    def set_vector_shape(self, x, y):
        x.set_shape([4, 128, 128])
        y.set_shape([128, 128])
        return x, y

    def normalize(self, x, y):
        x, y = x.numpy(), y.numpy()
        x = tf.keras.utils.normalize(x)
        return tf.convert_to_tensor(x, dtype=self.x_dtype), y

    def create_data_iterator(self, filenames_tensor, batch_size, repeat=False):
        """Create a Dataset iterator from a list of tfrecord files
        
        Parameters
        ----------
        filenames_tensor : list
            List of tfrecord file names to include in the dataset.
        batch_size : int
            Size of batches
        repeat : bool
            Whether to repeat the iteration over the dataset if the iterator runs out
            samples.
        """
        dataset = tf.data.TFRecordDataset(filenames_tensor)
        dataset = dataset.map(self.parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(lambda x, y: tf.py_function(self.normalize, [x, y], [self.x_dtype, self.y_dtype]))
        dataset = dataset.map(self.set_vector_shape)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        if repeat:
            dataset = dataset.repeat()
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

def test():

    records = glob('saved_dataset/**/**/*.tfrecord')
    recorder = TrainingRecorder(data_shape = [4, 128, 128])
    training_dataset = recorder.create_data_iterator(records, 1, repeat=False)
    for i, j in training_dataset:
        plt.imshow(i[0].numpy()[0])
        plt.savefig('./test/images/test.png')
        break