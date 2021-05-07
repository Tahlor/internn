import wandb
from subprocess import Popen

Popen("ssh -fNt -D 41080 tarch@login04.rc.byu.edu", shell=True)

# nano ../env/internn/lib/python3.8/socket.py
# import socks
# socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", 41080)
# socket = socks.socksocket

# 1. Start a new run
wandb.init(project="TEST")

# 2. Save model inputs and hyperparameters
config = wandb.config
config.dropout = 0.01


# wandb: Run `wandb offline` to turn off syncing.

