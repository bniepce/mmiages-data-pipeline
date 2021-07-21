import tensorflow as tf
import numpy as np

class TFRecorder(object):
    """
    Class to handle TFrecord creation
    ...

    Attributes
    ----------
    patch_dtype : str
        Type to use for patches array
    label_dtype: str
        Type to use for label array
    """
    def __init__(self):
        self.x_dtype = tf.float32
        self.y_dtype = tf.int8

    def _bytes_feature(self, value):
        """
        Tensorflow Feature helper function
        """
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(self, value):
        """
        Tensorflow Feature helper function
        """
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


class TrainingTFRecorder(TFRecorder):
    
    def __init__(self):
        super().__init__()
    
    def get_tf_example(self, x, y):
        """Convert data to tf.train.Example
    
        Parameters
        ----------
        x : np.array
            MRI data to save
        y : np.array
            Ground truth
            
        Return
        ----------
        example : tf.train.Example
            Tensorflow example to save as tfrecord
        """
        y = np.reshape(y, [y.shape[0]*y.shape[1]])
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': self._bytes_feature(x.tobytes()),
            'ground_truth': self._int64_feature(y)
        }))
        return example
    
    def save_tf_record(self, x, y, path):
        """Write tf.train.Example to tfrecord file
    
        Parameters
        ----------
        x : np.array
            MRI data to save
        y : np.array
            Ground truth
            
        Return
        ----------
        example : tf.train.Example
            Tensorflow example to save as tfrecord
        """
        with tf.io.TFRecordWriter(path) as writer:
            for i in range(len(x)):
                example = self.get_tf_example(x[i], y[i])
                writer.write(example.SerializeToString())
        