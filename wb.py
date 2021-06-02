import wandb


def socks():
    from subprocess import Popen
    #Popen("ssh -fNt -D 1080 tarch@login04.rc.byu.edu", shell=True)
    #Popen("ssh -fNt -D 1080 taylor@legentil", shell=True)
    import socks
    import socket
    from urllib import request
    from requests import utils
    socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", 1080)
    socket.socket = socks.socksocket

"""
nano socket.py

import socks
socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", 1080)
socket = socks.socksocket
"""
socks()

# 1. Start a new run
wandb.init(project="TEST")

# 2. Save model inputs and hyperparameters
config = wandb.config
config.dropout = 0.01


# wandb: Run `wandb offline` to turn off syncing.

#127.0.0.1 api.wandb.ai
#127.0.0.1 wandb.ai
#127.0.0.1 www.api.wandb.ai

from requests import Session

import sys
sys.exit()
