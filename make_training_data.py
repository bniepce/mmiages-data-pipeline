import os
from utils import io, parser

from pipeline import BraTSPipeline
from pipeline.processing import *
from pipeline.recorder import TrainingTFRecorder
from tqdm import tqdm


if __name__ == "__main__":

    
    """ INIT PARAMETERS """

    split_names = ['training', 'validation', 'testing']
    data_path = io.check_path('Please enter path to folder containing training and validation data folders :', '.mha', lvl=3)
    save_path = input('\033[94m'+'Please enter path to save dataset to : \n'+'\033[0m')
    print('\n')
        
    """ ORGANIZE BRATS DATA """
    
    io.organize_dataset(data_path)
    
    """ PERFORM PIPELINE """
    
    tfrecorder = TrainingTFRecorder()
    brats_pipeline = BraTSPipeline(data_path)
    brats_pipeline.add_operation(resize)
    brats_pipeline.add_operation(augment)
    
    n_cases = [len(glob(os.path.join(data_path, '{}/*'.format(name)))) for name in split_names]
    
    print('Number of cases : {}'.format(n_cases))
    
    for i, j in zip(split_names, n_cases):
        os.makedirs(os.path.join(save_path, i), exist_ok = True)                     
        with tqdm(total=j, desc = 'Applying pipeline to {} data :'.format(i)) as pbar:
            for idx, (x, y) in enumerate(brats_pipeline.process(i)):
                record_path = os.path.join(save_path, i, '{}_case_{}.tfrecord'.format(i, idx))
                tfrecorder.save_tf_record(x, y, record_path)
                pbar.update()