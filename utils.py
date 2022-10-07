import errno
import os.path

from easydict import Easydict as edict
from argparse import ArgumentParser
import shutil
import yaml



def command_line_parser(parser=None):
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument("--cfg", required=True,
                        help="config file for loading model/training model.")
    return parser


def read_yaml(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.load(f, yaml.Loader)

    return cfg


def save_yaml(config, fpath):
    try:
        with open(fpath, "w") as f:
            yaml.dump(config, f, default_flow_style=None)
    except Exception as e:
        print(e)


def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "."))


def get_abs_path(relative_path):
    proj_root = get_project_root()
    return os.path.abspath(os.path.join(proj_root, relative_path))


def load_configs(cfg_path):
    cfg = read_yaml(cfg_path)
    cfg = edict(cfg)
    return cfg


def safe_rmtree(paths=None):
    if paths:
        for path in paths:
            if os.path.isdir(path):
                try:
                    shutil.rmtree(path)
                except (IOError, OSError) as e:
                    if e.errno == errno.ENOENT:
                        pass
                    else:
                        raise


def safe_make_dir(a_dir, delete_existing=False):
    try:
        if delete_existing:
            safe_rmtree([a_dir])
        os.makedirs(a_dir)
    except (IOError, OSError) as e:
        if e.errno == errno.EEXIST and os.path.isdir(a_dir):
            pass
        else:
            raise


def convert_edict2dict(edict_obj):
    dict_obj = {}
    for key in edict_obj.keys():
        if isinstance(edict_obj[key], edict):
            dict_obj[key] = convert_edict2dict(edict_obj[key])
        elif isinstance(edict_obj[key], list):
            for idx, value in enumerate(edict_obj[key]):
                if isinstance(value, edict):
                    dict_obj.setdefault(key, []).append(convert_edict2dict(value))
                else:
                    dict_obj.setdefault(key, []).append(value)
        elif isinstance(edict_obj[key], dict):
            dict_obj[key] = convert_edict2dict(edict_obj[key])
        else:
            dict_obj[key] = edict_obj[key]
    return dict_obj
