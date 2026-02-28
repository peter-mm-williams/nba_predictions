"""Training callbacks for early stopping and checkpointing."""

from pathlib import Path

import torch
import torch.nn as nn

from src.utils.logging import get_logger

logger = get_logger("training.callbacks")


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement.
            min_delta: Minimum change to qualify as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def on_epoch_end(self, epoch: int, val_loss: float, model: nn.Module) -> bool:
        """Check if training should stop.

        Args:
            epoch: Current epoch number.
            val_loss: Validation loss for this epoch.
            model: The model being trained.

        Returns:
            True if training should stop.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False

        self.counter += 1
        if self.counter >= self.patience:
            logger.info(
                "EarlyStopping: no improvement for %d epochs (best: %.4f)",
                self.patience,
                self.best_loss,
            )
            return True
        return False


class ModelCheckpoint:
    """Save model checkpoint when validation loss improves."""

    def __init__(self, path: str, save_best_only: bool = True):
        """Initialize model checkpoint.

        Args:
            path: Path to save the checkpoint file.
            save_best_only: If True, only save when validation loss improves.
        """
        self.path = Path(path)
        self.save_best_only = save_best_only
        self.best_loss = float("inf")

    def on_epoch_end(self, epoch: int, val_loss: float, model: nn.Module) -> bool:
        """Save checkpoint if conditions are met.

        Args:
            epoch: Current epoch number.
            val_loss: Validation loss.
            model: Model to save.

        Returns:
            Always False (checkpointing doesn't stop training).
        """
        if not self.save_best_only or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"state_dict": model.state_dict(), "epoch": epoch, "val_loss": val_loss},
                self.path,
            )
            logger.info(
                "Checkpoint saved (epoch %d, val_loss: %.4f) to %s",
                epoch,
                val_loss,
                self.path,
            )
        return False


class CallbackList:
    """Container for multiple callbacks."""

    def __init__(self, callbacks: list):
        self.callbacks = callbacks

    def on_epoch_end(self, epoch: int, val_loss: float, model: nn.Module) -> bool:
        """Run all callbacks and return True if any requests stopping."""
        should_stop = False
        for cb in self.callbacks:
            if cb.on_epoch_end(epoch, val_loss, model):
                should_stop = True
        return should_stop
