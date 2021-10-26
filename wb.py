import wandb
from subprocess import Popen

# RUN wand_ prep.sh to edit defaulted installed SOCKS
# This might be to change the port to one that isn't managed by sudo

#kill_41080="ps -aux | grep "login04.rc.byu.edu" | awk "{print \$2}" | xargs kill -9"

open_port_command="""
server=${1:-login04.rc.byu.edu} # $(hostname)
if ! pgrep -f $server > /dev/null ; then
    ssh -fNt -D 41080 tarch@$server;
fi;
"""

Popen(open_port_command, shell=True)

# 1. Start a new run
wandb.init(project="TEST")

# 2. Save model inputs and hyperparameters
config = wandb.config
config.dropout = 0.01


# wandb: Run `wandb offline` to turn off syncing.

