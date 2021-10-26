import warnings
from pathlib import Path
import yaml
from itertools import product
import sys
sys.path.append("../slurm")
import gen
from subprocess import Popen

baseline_configs = ["00_master.yaml"]

variation_dict = {"experiment_type": ["vgg_embeddings","vgg_logits"],
                  "embedding_norm":["L2","softmax","default"]}

baseline_dict = {"max_intensity": 0}
baseline_dict = False

def cartesian_product(inp):
    return (dict(zip(inp.keys(), values)) for values in product(*inp.values()))

def check_yaml(yaml_dict):
    for key, value in yaml_dict.items():
        if value == 'null':
            warnings.warn("Detected 'null' as string; changing to None")
            yaml_dict[key] = None
        elif isinstance(value, str):
            yaml_dict[key] = get_int(value)
        elif isinstance(value, list):
            new_list = []
            for v in value:
                new_list.append(get_int(v))
            yaml_dict[key] = new_list
    return yaml_dict

def get_int(s):
    if isinstance(s, str):
        if s.lower() == 'true':
            return True
        elif s.lower() == 'false':
            return False
        else:
            try:
                return int(s)
            except ValueError:
                try:
                    return float(s)
                except ValueError:
                    return s
    else:
        return s

def replace_config(yaml_path, variation_list, new_folder="variants"):
    yaml_path = Path(yaml_path)
    yaml_file = check_yaml(yaml.load(Path(yaml_path).read_text(), Loader=yaml.SafeLoader))
    name = yaml_path.stem
    parent = yaml_path.parent
    ext = yaml_path.suffix

    output_dir = Path(parent / new_folder).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    output_files = []
    for variant in variation_list:
        new_yaml_file = yaml_file.copy()

        output_file = name
        for key, value in variant.items():
            new_yaml_file[key] = value
            file_name_variant_abbreviation = ""
            if isinstance(value, list):
                for val in value:
                    file_name_variant_abbreviation += Path(str(val)).stem + "_" # mostly get rid of "/" etc.
            else:
                file_name_variant_abbreviation += Path(str(value)).stem
            output_file += f"_{file_name_variant_abbreviation}"

        # Replace experiment output folder
        new_yaml_file["folder_outputs"] = yaml_file["folder_outputs"].replace("*EXPERIMENT*", f"variants/{output_file}")

        #with open((parent / name).with_suffix(ext), "w") as f:
        with (output_dir / (output_file + ext)).open(mode='w') as f:
            yaml.dump(new_yaml_file, f, default_flow_style=False, sort_keys=False)

        output_files.append(output_file)
    return output_files


def main(baseline_file, variation_dict, baseline_dict=None):
    all_combinations = list(cartesian_product(variation_dict))

    if baseline_dict:
        all_combinations += [baseline_dict]
        print(all_combinations)
    all_files = replace_config(baseline_file, all_combinations)
    print(all_files)

if __name__=="__main__":
    for config in baseline_configs:
        main(config, variation_dict, baseline_dict)

    gen.delete_old_sh()

    gen.Generator(group_path=Path("/lustre/scratch/grp/fslg_internn/"),
              environment=None,
              proj_dir=Path("./internn/lm"),
              cd_dir=None,
              python_script_path="train_BERT.py",
              config_root="./configs",
              sh_proj_root="./results",
              log_root=None,
              hardware_dict={"threads":7, "time":"4:00:00", "gpu":"pascal"},
              use_experiment_folders=True)

    # Make scripts executable
    Popen('find . -type f -iname "*.sh" -exec chmod +x {}  \;', shell=True)
    #gen.loop_configs(gen.config_root, gen.sh_root)
