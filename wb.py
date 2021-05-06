import wandb

# ~/.config/wandb/settings
# [default]
# #base_url = http://localhost:8080
# base_url = https://api.wandb.ai

# 1. Start a new run
#wandb.init(project="TEST")
wandb.init(project="TEST", config={"default":{"base_url":"https://127.0.0.1"}})

base_url='https://api.wandb.ai'

# 2. Save model inputs and hyperparameters
config = wandb.config
config.dropout = 0.01


# wandb: Run `wandb offline` to turn off syncing.