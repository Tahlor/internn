import os
import sys
from subprocess import Popen
from pathlib import Path
import socket

# GLOBALS
EMAIL = "taylornarchibald@gmail.com" # mikebbrodie@gmail.com
MEM = 100000

class Generator():
    def __init__(self,
                 group_path=Path("/lustre/scratch/grp/fslg_internn/"),
                 environment=None,
                 proj_dir = None,
                 cd_dir = None,
                 python_script_dir = None,
                 config_root = None,
                 sh_proj_root = None,
                 log_root = None,
                 hardware_dict=None,
                 *args, **kwargs):
        """

        Args:
            group_path: FSLG Group directory
            proj_dir: The main project folder -- EVERYTHING IS RELATIVE TO THIS
            cd_dir: The directory the python script should be run from (default: proj_dir)
            python_script_dir: The directory the python script is located (default: cd_dir)
            config_root: The directory with the desired configs
            sh_proj_root: Where to save the SH files (will mirror the config_root structure)
            log_root: Where to save the logs (default: sh_proj_root)
            *args:
            **kwargs:
        """

        self.group_path = group_path
        self.env = self.group_path / "env/internn" if environment is None else environment
        self.proj_dir = Path(self.group_path / "internn") if proj_dir is None else proj_dir
        self.cd_dir = self.proj_dir if cd_dir is None else cd_dir
        self.python_script_dir = self.python_script_dir if python_script_dir is None else python_script_dir

        # If running locally, setup path for files to be written to
        if socket.gethostname() in "Galois":
            self.proj_dir = Path("/media/data/GitHub") / proj_dir.relative_to(self.group_path)
            assert self.proj_dir.exists()

        # Paths relative to project directory
        self.config_root = proj_dir / "configs" if config_root is None else config_root # the config root folder
        self.sh_proj_root = self.sh_proj_root / "slurm/scripts" if sh_proj_root is None else sh_proj_root   # the .sh root folder
        self.log_root = self.sh_proj_root if log_root is None else log_root

        # Environment

        self.hardware_dict = {
            "default":{"threads":8, "time":"72:00:00", "gpu":"pascal"},
            } if hardware_dict is None else hardware_dict

        self.loop_configs(group_path,
                 environment,
                 proj_dir,
                 cd_dir,
                 python_script_dir,
                 config_root,
                 sh_proj_root)



    def gen(self, sh_path, log_path, env, command, hardware_dict, cd_path=None):

        time = hardware_dict["time"]
        threads = hardware_dict["threads"]
        mem = f"{int( MEM / threads )}MB"
        if cd_path is None:
            cd_path,_ = os.path.split(sh_path)
        self.mkdir(sh_path, parent=True)
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
    conda activate $PATH
    
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
        python_script_dir,
        config_root,
        sh_proj_root,
        log_root):

        p = Path(config_root)
        print(p)
        for config_path in p.rglob(f"*.yaml"):  # use rglob for recursion
            # Define paths
            subfolders = config_path.relative_to(p)  # = 'c/d'
            sh_path = Path(sh_proj_root / subfolders.parent / config_path.with_suffix('.sh').name)
            log_path = Path(log_root / subfolders.parent / ('log_' + config_path.with_suffix('.slurm').name))
            py_script = python_script_dir / "lm/train_BERT.py"
            cd_path = cd_dir
            command = f"python -u {py_script} --config '{config_path}'"
            #print(f"sh:{sh_path} log:{log_path} cd: {cd_path}")
            # if socket.gethostname() == "Galois":
            #     log_path = sh_proj_root / log_path.relative_to(proj_dir)
            #     cd_path = sh_proj_root / cd_dir.relative_to(proj_dir)
            self.gen(sh_path=sh_path, log_path=log_path, env=environment, command=command, hardware_dict=self.hardware_dict["default"], cd_path=cd_path)


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
    Generator(group_path=Path("/lustre/scratch/grp/fslg_internn/"),
             environment=None,
             proj_dir=Path("./internn/lm"),
             cd_dir=None,
             python_script_dir=None,
             config_root=".",
             sh_proj_root=None,
             log_root=None,
             hardware_dict=None)

    # Make scripts executable
    Popen('find . -type f -iname "*.sh" -exec chmod +x {}  \;', shell=True)