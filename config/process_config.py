import time
import itertools
import yaml
import shutil
import json
import os
from pathlib import Path
import sys
from easydict import EasyDict as edict
import logging

sys.path.append("..")
import internn_utils

logger = logging.getLogger("root."+__name__)

def find_config(config_name, config_root="./configs"):
    # Correct config paths
    if os.path.isfile(config_name):
        return config_name

    found_paths = []
    for d,s,fs in os.walk(config_root):
        for f in fs:
            if config_name == f:
                found_paths.append(os.path.join(d, f))

    # Error handling
    if len(found_paths) == 1:
        return found_paths[0]
    elif len(found_paths) > 1:
        raise Exception("Multiple {} config were found: {}".format(config_name, "\n".join(found_paths)))
    elif len(found_paths) < 1:
        raise Exception("{} config not found".format(config_name))

def fix_dict(d):
    for k in d:
        if isinstance(d[k], str) and d[k].lower() == "none":
            d[k] = None
        elif isinstance(d[k], dict):
            d[k] = fix_dict(d[k])
    return d

def incrementer(root, base, make_folder=True):
    new_folder = Path(root) / base
    increment = 0

    while new_folder.exists():
        increment += 1
        increment_string = f"{increment:02d}" if increment > 0 else ""
        new_folder = Path(root / (base + increment_string))

    new_folder.mkdir(parents=True, exist_ok=True)
    return new_folder


def read_config(config):
    config = Path(config)

    if config.suffix.lower() == ".json":
        return json.load(config.open(mode="r"))
    elif config.suffix.lower() == ".yaml":
        return internn_utils.fix_scientific_notation(yaml.load(config.open(mode="r"), Loader=yaml.Loader))
    else:
        raise "Unknown Filetype {}".format(config)

def load_config(config_path, hwr=True,
                testing=False,
                results_dir_override=None,
                subpath=None,
                create_logger=True):
    config_path = Path(config_path)
    project_path = Path(os.path.realpath(__file__)).parent.parent.absolute()
    logger.info("Project path", project_path)
    config_root = project_path / "configs"

    subpath = config_root / subpath if subpath else config_root


    # Path was specified, but not found
    # if config_root not in config_path.absolute().parents:
    #     raise Exception("Could not find config!")

    # Try adding a suffix
    if config_path.suffix != ".yaml":
        config_path = config_path.with_suffix(".yaml")

    # Go search for it
    if not config_path.exists():
        print(f"{config_path} does not exist")

        config_path = find_config(config_path.name, config_root=subpath)

    config = edict(read_config(config_path))
    config["name"] = Path(config_path).stem  ## OVERRIDE NAME WITH THE NAME OF THE YAML FILE
    config["project_path"] = project_path

    #config.counter = Counter()

    if results_dir_override:
        config.results_dir = results_dir_override

    if testing:
        config.TESTING = True

    # Fix
    config = fix_dict(config)

    if Path(config_path).stem=="RESUME":
        output_root = Path(config_path).parent
        experiment = config.experiment

        # Backup all_stats.json etc.
        backup = incrementer(output_root, "backup")
        print(backup)
        try:
            shutil.copy(output_root / "RESUME_model.pt", backup)
        except:
            pass
        for f in itertools.chain(output_root.glob("*.json"),output_root.glob("*.log")):
            shutil.copy(str(f), backup)

        output_root = output_root.as_posix()
    elif results_dir_override or ("results_dir_override" in config and config.results_dir_override):
        experiment = Path(results_dir_override).stem
        output_root = results_dir_override
    # If using preloaded model AND override is not mentioned
    elif (config["load_path"] and "results_dir_override" not in config):
        _output = incrementer(Path(config["load_path"]).parent, "new_experiment") # if it has a load path, create a new experiment in that same folder!
        experiment = _output.stem
        output_root = _output.as_posix()
    else:
        try: # Try to recreate the structure in the config directory (if this is in the config directory)
            experiment = Path(config_path).absolute().relative_to(Path(config_root).absolute()).parent
            if str(experiment) == ".": # if experiment is in root directory, use the experiment specified in the yaml
                experiment = config["experiment"]
            output_root = os.path.join(config["output_folder"], experiment)

        except Exception as e: # Fail safe; just dump to "./output/CONFIG NAME"
            logger.warning(f"Failed to find relative path of config file {config_root} {config_path}")
            experiment = Path(config_path).stem
            output_root = os.path.join(config["output_folder"], experiment)

    config.output_root = output_root

    # Use config folder to determine output folder
    config["experiment"] = str(experiment)
    logger.info(f"Experiment: {experiment}, Results Directory: {output_root}")

    hyper_parameter_str='{}'.format(
         config["name"],
     )

    train_suffix = '{}-{}'.format(
        time.strftime("%Y%m%d_%H%M%S"),
        hyper_parameter_str)

    if config["SMALL_TRAINING"] or config["TESTING"]:
        train_suffix = "TEST_"+train_suffix

    config["full_specs"] = train_suffix

    # Directory overrides
    if 'results_dir' not in config.keys():
        config['results_dir']=os.path.join(output_root, train_suffix)
    if 'output_predictions' not in config.keys():
        config['output_predictions']=False
    if "log_dir" not in config.keys():
        config["log_dir"]=os.path.join(output_root, train_suffix)
    if "image_dir" not in config.keys():
        config["image_dir"] = os.path.join(config['results_dir'], "imgs")

    if hwr:
        config["image_test_dir"] = os.path.join(config["image_dir"], "test")
        config["image_train_dir"] = os.path.join(config["image_dir"], "train")

    # Create paths
    for path in [output_root] + [config[d] for d in config.keys() if "_dir" in d]:
        if path is not None and len(path) > 0 and not os.path.exists(path):
            Path(path).mkdir(parents=True, exist_ok=True)

    # Make a link to most recent run
    try:
        link = "./RECENT.lnk"
        old_link = "./RECENT.lnk2"
        if os.path.exists(old_link):
            os.remove(old_link)
        if os.path.exists(link):
            os.rename(link, old_link)
        #symlink(config['results_dir'], link)
    except Exception as e:
        logger.warning("Problem with RECENT link stuff: {}".format(e))

    # Copy config to output folder
    #parent, child = os.path.split(config)
    try:
        shutil.copy(config_path, config['results_dir'])
    except Exception as e:
        logger.info(f"Could not copy config file: {e}")

    for root in ["training_root", "testing_root"]:
        if root in config and config[root].find("data")==0:
            config[root] = Path(config.project_path) / config.training_root

    logger.info(f"Using config file: {config_path}")
    config["logger"] = logger
    config["stats"] = {}

    if not config.gpu_if_available:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    return config
