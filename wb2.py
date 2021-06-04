import wandb

def socks():
    from subprocess import Popen
    #Popen("ssh -fNt -D 35000 tarch@login04.rc.byu.edu", shell=True)
    import socks
    import socket
    from urllib import request
    import urllib3
    import requests
    socks.set_default_proxy(socks.SOCKS5, "localhost", 35000)
    socket.socket = socks.socksocket

# 1. Start a new run
wandb.init(project="TEST")

# 2. Save model inputs and hyperparameters
config = wandb.config
config.dropout = 0.01


# wandb: Run `wandb offline` to turn off syncing.
