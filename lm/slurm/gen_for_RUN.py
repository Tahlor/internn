import gen
from subprocess import Popen
from pathlib import Path

if __name__=="__main__":
    gen.delete_old_sh()
    g = gen.Generator(group_path=Path("/lustre/scratch/grp/fslg_internn/"),
              environment=None,
              proj_dir=Path("./internn/lm"),
              cd_dir=None,
              python_script_path="train_BERT.py",
              config_root="./results",
              sh_proj_root="./results",
              log_root=None,
              hardware_dict=None,
              use_experiment_folders=True)

    # Make scripts executable
    output = g.sh_proj_root
    Popen('cd '+ str(output) + ' ; find . -type f -iname "*.sh" -exec chmod +x {}  \;', shell=True)
