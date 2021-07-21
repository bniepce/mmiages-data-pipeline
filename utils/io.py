import os, random, shutil
import numpy as np
from glob import glob


def check_file(inp_str : str, ext : str) -> str:
    """Ask the user to enter a file path.
    Check if input string matches the existence of a file with a given extension.
    The input is asked until a correct file is given.
    
    Parameters
    ----------
    inp_str : str
        String the check
    ext : 
        Extension of the file
    Return
    ----------
    f : str
        String name of a file
    """
    while True:
        try:
            f = input('\033[94m'+inp_str+'\n'+'\033[0m')
            if os.path.isfile(f):
                if f.endswith(ext):
                    break
                else:
                    raise
            else:
                raise
        except:
            print('\033[93m'+'Not a correct file. Try again.'+'\033[0m')
            continue
        break
    return f

def check_path(inp_str : str, ext : str, lvl : int = 0) -> str:
    """Ask the user to enter data path.
    Check a path by looking if files with given extension are present 
    at a certain level in the folder tree. The input is asked until files with the right extensions are found.
    
    Parameters
    ----------
    inp_str : str
        Path string to check
    ext : 
        Extension of files to look for
    lvl : 
        Level in the folder tree in which to look for the files
    Return
    ----------
    path : str
    """
    while True:
        try:
            path = input('\033[94m'+inp_str+'\n'+'\033[0m')
            prefix = lvl*'**/'
            if glob(path+prefix+'*{}'.format(ext)):
                break
            else:
                raise
        except:
            print('\033[93m'+'No {} file found. Try again.'.format(ext)+'\033[0m')
            continue
        break
    return path

def organize_dataset(data_path):
    
    split_names = ['training', 'validation', 'testing']
    
    if os.path.isdir(data_path + 'training'):
        print('Skipping dataset splitting as it has already been done.')
    else:
        def split(l, perc):
            splits = np.cumsum(perc)/100.
            splits = splits[:-1]
            splits *= len(l)
            splits = splits.round().astype(np.int)
            return np.split(l, splits)

        for i in split_names:
            os.makedirs(os.path.join(data_path, i), exist_ok = True)

        HGG = glob(os.path.join(data_path, 'HGG/*'))
        LGG = glob(os.path.join(data_path, 'LGG/*'))
        
        hgg_split, lgg_split = split(HGG, [70, 20, 10]), split(LGG, [70, 20, 10])
        
        for (s_hgg, s_lgg, s_name) in zip(hgg_split, lgg_split, split_names):
            for case in s_hgg:
                case_name = os.path.basename(os.path.normpath(case))
                shutil.move(case, os.path.join(data_path, s_name, case_name))
                
            for case in s_lgg:
                case_name = os.path.basename(os.path.normpath(case))
                shutil.move(case, os.path.join(data_path, s_name, case_name))
                
        shutil.rmtree(os.path.join(data_path, 'HGG'))
        shutil.rmtree(os.path.join(data_path, 'LGG'))
        
        print('Dataset reorganized with success.')
    
    return