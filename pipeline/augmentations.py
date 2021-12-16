import tensorlayer as tl
import numpy as np

def rotation(image_list):
    image_list = tl.prepro.rotation_multi(image_list, rg=130, is_random=False, fill_mode='nearest')
    return image_list

def shift(image_list):
    image_list = tl.prepro.shift_multi(image_list, wrg=0.10, hrg=0.10, is_random=True, fill_mode='constant')
    return image_list

def elastic_transform(image_list):
    image_list = tl.prepro.elastic_transform_multi(image_list,alpha=720, sigma=24, is_random=True)
    return image_list

def shear(image_list):
    image_list = tl.prepro.shear_multi(image_list, 0.05,is_random=True, fill_mode='constant')
    return image_list

def flip(image_list):
    image_list = tl.prepro.flip_axis_multi(image_list,axis=1, is_random=True)
    return image_list

aug_operations = [rotation, flip, shear, shift, elastic_transform]