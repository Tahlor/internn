import shutil
import os
import sys
from subprocess import Popen
from pathlib import Path
import socket



# GLOBALS
EMAIL = "taylornarchibald@gmail.com" # mikebbrodie@gmail.com
MEM = 100000
GALOIS_GROUP_PATH = "/media/data/GitHub"
class Generator():
    def __init__(self,
                 group_path,
                 proj_dir,
                 python_script_path,
                 environment=None,
                 cd_dir=None,
                 config_root=None,
                 sh_proj_root=None,
                 log_root=None,
                 hardware_dict=None,
                 use_experiment_folders=False,
                 *args, **kwargs):
        """

        Args:
            group_path: FSLG Group directory -- SHOULD MIRROR Github folder on local machine
            proj_dir: The main project folder -- EVERYTHING IS RELATIVE TO THIS EXCEPT ENVIRONMENT
            python_script_path: The path the python script is located (NO DEFAULT)
            environment: RELATIVE TO GROUP_PATH
            cd_dir: The directory the python script should be run from (default: proj_dir)
            config_root: The directory with the desired configs
            sh_proj_root: Where to save the SH files (will mirror the config_root structure); needs to be on GALOIS if running from GALOIS
            log_root: Where to save the logs (default: sh_proj_root)
            hardware_dict (string / dict): Either a dictionary defining attributes or a string to specify a default one
            use_experiment_folders (bool): Create folders for logs and scripts to go (rather than naming them).
            *args:
            **kwargs:
            config_root_REFERENCE = WHERE TO REFERENCE CONFIGS ON FSL -- NEEDS TO BE FSL
            config_root_SEARCH = WHERE TO SEARCH FOR CONFIGS -- NEEDS TO BE LOCAL (on GALOIS)
        """
        self.use_experiment_folders=use_experiment_folders
        self.group_path = self.current_group_path = group_path
        self.env = self.group_path / "env/internn" if environment is None else environment
        self.proj_dir = Path(self.group_path / "internn") if proj_dir is None else self.group_path / proj_dir
        self.python_script_path = Path(self.proj_dir) / python_script_path
        self.cd_dir = Path(self.python_script_path).parent if cd_dir is None else self.proj_dir / cd_dir

        # Paths relative to project directory
        self.config_root_REFERENCE = self.config_root_SEARCH = proj_dir / "configs" if config_root is None else self.proj_dir / config_root # the config root folder
        self.sh_proj_root = self.cd_dir / "slurm/scripts" if sh_proj_root is None else self.proj_dir / sh_proj_root   # the .sh root folder
        self.log_root = self.sh_proj_root if log_root is None else log_root

        # If running locally, setup path for files to be written to
        if socket.gethostname() in "Galois":
            self.sh_proj_root = Path(GALOIS_GROUP_PATH) / self.sh_proj_root.relative_to(self.group_path)
            self.config_root_SEARCH = Path(GALOIS_GROUP_PATH) / Path(self.config_root_REFERENCE).relative_to(self.group_path)
            self.current_group_path = Path(GALOIS_GROUP_PATH)
            assert self.sh_proj_root

        # Environment
        self.hardware_dict = {
            "default":{"threads":7, "time":"72:00:00", "gpu":"pascal"},
            "custom": hardware_dict}

        if not hardware_dict:
            hw="default"
        elif isinstance(hardware_dict, dict):
            hw="custom"
        else:
            hw=hardware_dict

        self.loop_configs(group_path=self.group_path,
                          environment=self.env,
                          proj_dir=self.proj_dir,
                          cd_dir=self.cd_dir,
                          python_script_path=self.python_script_path,
                          config_root_SEARCH=self.config_root_SEARCH,
                          config_root_REFERENCE=self.config_root_REFERENCE,
                          sh_proj_root=self.sh_proj_root,
                          log_root=self.log_root,
                          hw=hw)



    def gen(self, sh_path, log_path, env, command, hardware_dict, cd_path=None):

        time = hardware_dict["time"]
        threads = hardware_dict["threads"]
        mem = f"{int( MEM / threads )}MB"
        if cd_path is None:
            cd_path,_ = os.path.split(sh_path)
        sh_path.parent.mkdir(exist_ok=True, parents=True)
        print(f"Writing {sh_path}")
        with open(f"{sh_path}", "w") as f:
                f.write(f"""#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -C 'rhel7&pascal'
#SBATCH --mem-per-cpu {mem}
#SBATCH --ntasks {threads}
#SBATCH --nodes=1
#SBATCH --output="{log_path}"
#SBATCH --time {time}
#SBATCH --mail-user={EMAIL}   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#%Module

module purge
module load cuda/10.1
module load cudnn/7.6

export PATH="{env}:$PATH"
eval "$(conda shell.bash hook)"
conda activate {env}

cd "{cd_path}"
which python
{command}
    """)

    #def loop_configs(self, config_root, script_root, ext=".yaml"):
    def loop_configs(self,
                     group_path,
                     environment,
                     proj_dir,
                     cd_dir,
                     python_script_path,
                     config_root_SEARCH,
                     config_root_REFERENCE,
                     sh_proj_root,
                     log_root,
                     hw):

        p = Path(config_root_SEARCH)
        print(p)
        for config_path in p.rglob(f"*.yaml"):  # use rglob for recursion
            # Define paths
            fsl_config_path = Path(config_root_REFERENCE) / config_path.relative_to(Path(config_root_SEARCH)) # FSL
            subfolders = config_path.relative_to(p)  # = 'c/d'
            # Create a subfolder named after the config
            if config_path.parent.stem != subfolders.parent.stem:
                sh_path = Path(sh_proj_root / subfolders.parent / config_path.with_suffix('.sh').name) # where the .sh is saved, LOCAL
            else:
                sh_path = Path(sh_proj_root / config_path.with_suffix('.sh').name)  # where the .sh is saved, LOCAL

            log_path = Path(log_root / subfolders.parent / ('log_' + config_path.with_suffix('.slurm').name)) # log path, FSL
            if self.use_experiment_folders:
                experiment_folder = Path(sh_proj_root / subfolders.parent / config_path.stem); experiment_folder.mkdir(parents=True,exist_ok=True)
                sh_path = experiment_folder / config_path.with_suffix('.sh').name
                fsl_experiment_folder = group_path / Path(experiment_folder).relative_to(self.current_group_path)
                fsl_config_path = fsl_experiment_folder / config_path.name
                log_path =  fsl_experiment_folder / ('log.slurm')
                #os.rename(config_path, experiment_folder / config_path.name)
                shutil.copy(config_path, experiment_folder / config_path.name)

            # Make config relative to CD dir; (log dir would need to be relative to CWD which we don't know)
            fsl_config_path = "." / fsl_config_path.relative_to(cd_dir)


            py_script = python_script_path
            cd_path = cd_dir
            command = f"python -u {py_script} '{fsl_config_path}'"
            #print(f"sh:{sh_path} log:{log_path} cd: {cd_path}")
            # if socket.gethostname() == "Galois":
            #     log_path = sh_proj_root / log_path.relative_to(proj_dir)
            #     cd_path = sh_proj_root / cd_dir.relative_to(proj_dir)
            self.gen(sh_path=sh_path, log_path=log_path, env=environment, command=command, hardware_dict=self.hardware_dict[hw], cd_path=cd_path)


def loop_exp(config_folder):
    # Loop over all experiment variations
    pass

def get_sh(path, ext=".sh"):
    for ds, s, fs in os.walk(path):
        if ds.lower() in ("old", "~archive"):
            continue
        for f in fs:
            if f[-len(ext):] == ext:
                yield os.path.join(ds, f)

def is_iterable(object, string_is_iterable=True):
    """Returns whether object is an iterable. Strings are considered iterables by default.

    Args:
        object (?): An object of unknown type
        string_is_iterable (bool): True (default) means strings will be treated as iterables
    Returns:
        bool: Whether object is an iterable

    """

    if not string_is_iterable and type(object) == type(""):
        return False
    try:
        iter(object)
    except TypeError as te:
        return False
    return True

def mkdir(paths, parent=False):
    if not is_iterable(paths):
        paths = paths,
    for path in paths:
        if parent:
            path, _ = os.path.split(path)
        if path is not None and len(path) > 0 and not os.path.exists(path):
            os.makedirs(path)


def delete_old_sh(path="."):
    for sh in get_sh(path):
        os.remove(sh)

if __name__=="__main__":
    delete_old_sh()
    g = Generator(group_path=Path("/lustre/scratch/grp/fslg_internn/"),
              environment=None,
              proj_dir=Path("./internn/lm"),
              cd_dir=None,
              python_script_path="train_BERT.py",
              config_root="./configs",
              sh_proj_root="./results",
              log_root=None,
              hardware_dict=None,
              use_experiment_folders=True)

    # Make scripts executable
    output = g.sh_proj_root
    Popen('cd '+ str(output) + ' ; find . -type f -iname "*.sh" -exec chmod +x {}  \;', shell=True)
