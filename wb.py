import wandb


def socks():
    from subprocess import Popen
    #Popen("ssh -fNt -D 1080 tarch@login04.rc.byu.edu", shell=True)
    import socks
    import socket
    from urllib import request
    from urllib3 import request
    import requests, urllib3, urllib
    socks.set_default_proxy(socks.SOCKS5, "localhost", 1080)
    socket.socket = socks.socksocket

socks()

# 1. Start a new run
wandb.init(project="TEST")

# 2. Save model inputs and hyperparameters
config = wandb.config
config.dropout = 0.01


# wandb: Run `wandb offline` to turn off syncing.