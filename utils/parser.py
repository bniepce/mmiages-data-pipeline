import yaml

def yaml_parser(file):
    """Parse a yaml file and returns it as a dict
    
    Parameters
    ----------
    file : str
        Path to .yaml file
    Return
    ----------
    yaml_dict : dict
    """
    with open(file, 'r') as ymlfile:
        yaml_dict = yaml.load(ymlfile, yaml.FullLoader)
    return yaml_dict