"""
scripts/train_hgvfi.py
----------------------
Entry-point script: train the HintGuidedVFI model on Vimeo-90K.

Usage
-----
    python scripts/train_hgvfi.py [--epochs 30] [--batch-size 4] [--limit 4000]

All hyperparameters are configurable via command-line flags.
Model checkpoints are saved every 5 epochs; the final weights go to MODEL_PATH.
"""

import argparse
import os

import torch

from hgvfi.data import load_vimeo_triplet
from hgvfi.models import HintGuidedVFI
from hgvfi.train import run_training
from hgvfi.visualize import plot_training_history


# ---------------------------------------------------------------------------
# Paths (edit or override via CLI)
# ---------------------------------------------------------------------------

DATASET_ROOT = "vimeo_triplet/"
TRAIN_LIST   = DATASET_ROOT + "tri_trainlist.txt"
TEST_LIST    = DATASET_ROOT + "tri_testlist.txt"
MODEL_PATH   = DATASET_ROOT + "hgvfi_model.pt"
PROJECT_ROOT = "."


def parse_args():
    parser = argparse.ArgumentParser(description="Train HintGuidedVFI on Vimeo-90K")
    parser.add_argument("--epochs",     type=int,   default=30,    help="Number of training epochs")
    parser.add_argument("--batch-size", type=int,   default=4,     help="Training batch size")
    parser.add_argument("--lr",         type=float, default=2e-4,  help="Initial learning rate")
    parser.add_argument("--limit",      type=int,   default=None,  help="Limit dataset size (debug)")
    parser.add_argument("--dataset",    type=str,   default=DATASET_ROOT)
    parser.add_argument("--model-path", type=str,   default=MODEL_PATH)
    return parser.parse_args()


def main():
    args = parse_args()

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    train_list = os.path.join(args.dataset, "tri_trainlist.txt")
    test_list  = os.path.join(args.dataset, "tri_testlist.txt")

    train_paths = load_vimeo_triplet(args.dataset, train_list)
    test_paths  = load_vimeo_triplet(args.dataset, test_list)
    print(f"Training triplets : {len(train_paths):,}")
    print(f"Test triplets     : {len(test_paths):,}")

    model = HintGuidedVFI().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters  : {total_params:,}")

    history = run_training(
        train_paths,
        test_paths,
        model,
        device,
        project_root=PROJECT_ROOT,
        model_path=args.model_path,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        limit=args.limit,
    )

    fig = plot_training_history(history)
    fig.savefig("training_history.png", bbox_inches="tight")
    print("Training history saved to training_history.png")


if __name__ == "__main__":
    main()
