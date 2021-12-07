import os, argparse
from utils import io, parser

from pipeline import BraTSPipeline
from pipeline.processing import *
from pipeline.h5 import HDF5Store
from pipeline.recorder import TrainingTFRecorder
from tqdm import tqdm


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        help='Path to data directory',
        required=True
    )
    parser.add_argument(
        '--save_path',
        help='Path to save data',
        required=True
    )
    parser.add_argument(
        '--format',
        help='Format to save data',
        required=True
    )

    args = parser.parse_args()
    
    """ INIT PARAMETERS """

    split_names = ['training', 'testing']
    data_path = io.check_path(args.data_path, '.mha', lvl=3)
    save_path = args.save_path
    save_format = args.format
    print('\n')

    if save_format not in ['tfrecord', 'npy', 'h5']:
        raise ValueError('Format has to be in '.format(['tfrecord', 'npy', 'h5']))
        
    """ ORGANIZE BRATS DATA """
    
    io.organize_dataset(data_path, split_names)
    
    """ PERFORM PIPELINE """
    
    tfrecorder = TrainingTFRecorder()
    brats_pipeline = BraTSPipeline(data_path)
    brats_pipeline.add_operation(resize)
    # brats_pipeline.add_operation(normalize)
    # brats_pipeline.add_operation(augment)
    
    n_cases = [len(glob(os.path.join(data_path, '{}/*'.format(name)))) for name in split_names]
    
    print('Number of cases : {}'.format(n_cases))
    
    for i, j in zip(split_names, n_cases):
        os.makedirs(os.path.join(save_path, i), exist_ok = True) 
        
        if save_format == 'h5':
            file_path = os.path.join(save_path, i, 'h5_dataset.h5')
            h5_store = HDF5Store(file_path, ['X', 'Y'], shapes=[(128, 128, 4), (128, 128)], dtype=[np.uint8, np.float32])
        
        with tqdm(total=j, desc = 'Applying pipeline to {} data :'.format(i)) as pbar:

            for idx, (x, y) in enumerate(brats_pipeline.process(i)):
                if save_format == 'tfrecord':
                    file_path = os.path.join(save_path, i, '{}_case_{}.tfrecord'.format(i, idx))
                    tfrecorder.save_tf_record(x, y, file_path)

                elif save_format == 'npy':
                    os.makedirs(os.path.join(save_path, i, 'x'), exist_ok=True)
                    os.makedirs(os.path.join(save_path, i, 'y'), exist_ok=True)
                    np.save(os.path.join(save_path, i, 'x', '{}_case_X_{}.npy'.format(i, idx)), x)
                    np.save(os.path.join(save_path, i, 'y', '{}_case_Y_{}.npy'.format(i, idx)), y)

                elif save_format == 'h5':
                    for data, target in zip(x, y):
                        data *= 255.0 / data.max()
                        target = np.divide(target.astype('float32'), 4)
                        data = np.moveaxis(data, 0, 2)
                        h5_store.append('X', data.astype('uint8'), data.shape)
                        h5_store.append('Y', target, target.shape)

                pbar.update()