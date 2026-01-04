#!/bin/bash
# =============================================================================
# Sync wandb offline runs to wandb server
# Run this script in AIcoder (with internet access)
# =============================================================================

echo "Syncing wandb offline runs..."
echo "=============================="

# Sync all offline runs
if [ -d "./wandb" ]; then
    wandb sync ./wandb/
    echo "Sync completed!"
else
    echo "No wandb directory found. Make sure you're in the correct directory."
fi
