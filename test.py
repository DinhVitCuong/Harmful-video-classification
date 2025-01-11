import wandb
import random
from tqdm import tqdm  # Import tqdm for the progress bar

# Start a new wandb run to track this script
wandb.init(
    # Set the wandb project where this run will be logged
    project="my-awesome-project",

    # Track hyperparameters and run metadata
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    },
)

# Simulate training
epochs = 10
offset = random.random() / 5

# Use tqdm for the progress bar
for epoch in tqdm(range(2, epochs), desc="Training Progress"):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset

    # Log metrics to wandb
    wandb.log({"epoch": epoch, "accuracy": acc, "loss": loss})

# [Optional] Finish the wandb run, necessary in notebooks
wandb.finish()
