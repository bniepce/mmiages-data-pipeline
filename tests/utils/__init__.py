import tensorflow as tf
from abc import ABC, abstractmethod

class AbstractRecorder(ABC):
    """Class to read tfrecord and create tensorflow datasets
    ...
    Attributes
    ----------
    x_shape : tuple 
        Shape of x registered in tfrecord files
    y_shape : tuple 
        Shape of y registered in tfrecord files
    x_dtype : str
        Type to use for data array
    y_dtype: str
        Type to use for label array
    """
    def __init__(self, x_shape, y_shape, x_dtype = tf.float32, y_dtype = tf.uint8):
        super().__init__()
        self.x_shape = x_shape
        self.y_shape = y_shape
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
    x_shape : tuple 
        Shape of x registered in tfrecord files
    y_shape : tuple 
        Shape of y registered in tfrecord files
    x_dtype : str
        Type to use for data array
    y_dtype: str
        Type to use for label array
    """
    def __init__(self, x_shape, y_shape, x_dtype = tf.float32, y_dtype = tf.uint8):
        super().__init__(x_shape, y_shape, x_dtype, y_dtype)

    def parse_function(self, proto):
        """Parse function to create the training Dataset object
        
        Parameters
        ----------
        proto : str
            A scalar string Tensor, single serialized tf Example.
        """
        features = {
            'image': tf.io.FixedLenFeature([], dtype=tf.string),
            'ground_truth': tf.io.FixedLenFeature(self.y_shape, dtype=tf.int64)
        }
        record = tf.io.parse_single_example(proto, features)
        image_raw = tf.io.decode_raw(record['image'], self.x_dtype)
        image_raw = tf.reshape(image_raw, shape=self.x_shape)

        label_raw = tf.one_hot(record["ground_truth"], 5, dtype=tf.int32)
        label_raw = tf.cast(label_raw, self.y_dtype)
        return image_raw, label_raw

    def set_vector_shape(self, x, y):
        x.set_shape(self.x_shape)
        y.set_shape(self.y_shape.append(5))
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