import os, cv2
import numpy as np
import SimpleITK as sitk
from .augmentations import aug_operations
from glob import glob

def get_scans(source : list) -> list:
    """Yield MRI sequence file path from all cases of the BraTS 2015 dataset
    
    Parameters
    ----------
    source : list
        List of MRI case
        
    Yield
    ----------
    scans : list
        List of .mha file containing all 4 sequences of one MRI case
    """
    for case in source:
        scans = glob(os.path.join(case, '**/*.mha*'))
        scans.sort()
        yield scans
        
def convert_scans(source : list) -> tuple:
    """Convert MRI data into an np.array
    by reading .mha files and deleting empty slices.
    
    Parameters
    ----------
    source : list
        List of .mha file to convert to numpy arrays
        
    Yield
    ----------
    data : tuple
        Data tuple containing MRI array of shape (4, number_of_slices, width, height) and 
        Ground truth array of shape (number_of_slices, width, height)
    """
    for scan in source:
        dataset = np.array([sitk.GetArrayFromImage(sitk.ReadImage(s)) for s in scan])
        x = np.squeeze(dataset[:len(dataset)-1])
        y = dataset[-1]
        non_zero_indices = list(set(np.nonzero(y)[0]))
        y = y[list(non_zero_indices)]
        x = np.array([i[list(non_zero_indices)] for i in x])
        yield (np.moveaxis(x.astype('float32'), 0, 1), y)

def resize(source : tuple, x_bound : int = 56, y_bound : int = 184) -> tuple:
    """Resize slices of dataset X and ground truth Y.
    By default slices are cropped from the bounds 56 and 184 giving slices of dimension 128x128.
    
    Parameters
    ----------
    data : tuple
        Data tuple containing MRI array of shape (4, number_of_slices, width, height) and 
        Ground truth array of shape (number_of_slices, width, height)
    x_bound : int
        X axis bound. Default to 56
    y_bound : int
        Y axis bound. Default to 184
    Return
    ----------
    resized_data : tuple
        Resized dataset
    """
    for data in source:
        x_resized = []
        x, y = data
        for sequence in x:
            x_resized.append(np.array([slc[x_bound:y_bound, x_bound:y_bound] for slc in sequence]))
        y_resized = np.array([slc[x_bound:y_bound, x_bound:y_bound] for slc in y])

        yield (np.array(x_resized), y_resized)

def augment(source : tuple, augmentations = aug_operations, augment_ratio=0.1):
    """Apply augmentation operations to a ratio of the MRI data and ground truth.
    
    Parameters
    ----------
    source : tuple
        Data tuple containing MRI array of shape (4, number_of_slices, width, height) and 
        Ground truth array of shape (number_of_slices, width, height)
    augmentations : list
        List of operations to perform
    augment_ratio : float, In [0, 1].
        Ratio of the data to augment. Default to 0.30.
        
    Return
    ----------
    augmented_data : tuple
        Augmented data
    """
    for data in source:
        x, y = data
        aug_indices = np.random.randint(x.shape[0], size= int(x.shape[0] * augment_ratio))
        augmented_slices = np.zeros((1, x.shape[1] + 1, x.shape[2], x.shape[3]))
        for idx, (x_slc, y_slc) in enumerate(zip(x[aug_indices], y[aug_indices])):
            x_slc, y_slc = x_slc[..., np.newaxis], y_slc[np.newaxis, ..., np.newaxis]
            for f in augmentations:
                x_y_augment = np.concatenate([x_slc, y_slc])
                x_y_augment = np.squeeze(f(x_y_augment))
                augmented_slices = np.concatenate([augmented_slices, x_y_augment[np.newaxis, ...]])
        yield (np.concatenate([x, augmented_slices[:,:-1]]).astype('float32'), 
               np.concatenate([y, augmented_slices[:,-1]]).astype('int8'))

def normalize(source : tuple):
    for data in source:
        x, y = data
        print(x.shape)
        x *= 255.0 / x.max()
        y = np.divide(y.astype('uint8'), 4)
        x = np.moveaxis(x, 1, 3)
        yield (x.astype('uint8'), y)