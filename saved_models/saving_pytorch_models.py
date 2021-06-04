import glob
from pathlib import Path
import json
import warnings
import os
import torch

def dict_to_list(d):
    idx_to_char = []
    for i in range(0, max(d.keys())+1):
        idx_to_char.append(d[i])
    return idx_to_char

def load_model(config):
    # User can specify folder or .pt file; other files are assumed to be in the same folder
    if os.path.isfile(config["load_path"]):
        old_state = torch.load(config["load_path"])
        path, child = os.path.split(config["load_path"])
    else:
        old_state = torch.load(os.path.join(config["load_path"], "baseline_model.pt"))
        path = config["load_path"]

    for key in ["idx_to_char", "char_to_idx"]:
        if key in old_state.keys():
            if key == "idx_to_char":
                old_state[key] = dict_to_list(old_state[key])
            config[key] = old_state[key]

    if "model" in old_state.keys():
        config["model"].load_state_dict(old_state["model"])
        if "optimizer" in config.keys():
            config["optimizer"].load_state_dict(old_state["optimizer"])
        config["global_counter"] = old_state["global_step"]
        config["starting_epoch"] = old_state["epoch"]
        config["current_epoch"] = old_state["epoch"]
    else:
        config["model"].load_state_dict(old_state)

    # Launch visdom
    if config["use_visdom"]:
        try:
            config["visdom_manager"].load_log(os.path.join(path, "visdom.json"))
        except:
            warnings.warn("Unable to load from visdom.json; does the file exist?")
            ## RECREATE VISDOM FROM FILE IF VISDOM IS NOT FOUND


    except:
        warnings.warn("Could not load from all_stats.json")


def save_model(config, bsf=False):
    # Can pickle everything in config except items below
    # for x, y in config.items():
    #     print(x)
    #     if x in ("criterion", "visdom_manager", "trainer"):
    #         continue
    #     torch.save(y, "TEST.pt")

    # Save the best model
    if bsf:
        path = os.path.join(config["results_dir"], "BSF")
        mkdir(path)
    else:
        path = config["results_dir"]

    #    'model_definition': config["model"],
    state_dict = {
        'epoch': config["current_epoch"] + 1,
        'model': config["model"].state_dict(),
        'optimizer': config["optimizer"].state_dict(),
        'global_step': config["global_step"],
        "idx_to_char": config["idx_to_char"],
        "char_to_idx": config["char_to_idx"]
    }

    config["main_model_path"] = os.path.join(path, "{}_model.pt".format(config['name']))
    torch.save(state_dict, config["main_model_path"])

    if "nudger" in config.keys():
        state_dict["model"] = config["nudger"].state_dict()
        torch.save(state_dict, os.path.join(path, "{}_nudger_model.pt".format(config['name'])))

    save_stats(config, bsf)

    # Save visdom
    if config["use_visdom"]:
        try:
            path = os.path.join(path, "visdom.json")
            config["visdom_manager"].save_env(file_path=path)
        except:
            warnings.warn(f"Unable to save visdom to {path}; is it started?")
            config["use_visdom"] = False

    # Copy BSF stuff to main directory
    if bsf:
        for filename in glob.glob(path + r"/*"):
            shutil.copy(filename, config["results_dir"])

    if config["save_count"]==0:
        create_resume_training(config)
    config["save_count"] += 1
