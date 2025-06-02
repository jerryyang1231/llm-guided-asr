import wandb
wandb.init(project="test", settings=wandb.Settings(init_timeout=10))
wandb.finish()
