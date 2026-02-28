"""Training loop orchestration for PyTorch models."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.sequences import NBASequenceDataset
from src.training.callbacks import CallbackList, EarlyStopping, ModelCheckpoint
from src.utils.logging import get_logger

logger = get_logger("training.trainer")


class Trainer:
    """Orchestrates training for PyTorch-based sequential models."""

    def __init__(self, config: dict):
        """Initialize the trainer.

        Args:
            config: Training configuration section.
        """
        self.epochs = config.get("epochs", 100)
        self.batch_size = config.get("batch_size", 64)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.weight_decay = config.get("weight_decay", 0.0001)
        self.gradient_clip_norm = config.get("gradient_clip_norm", 1.0)
        self.device = torch.device(config.get("device", "cpu"))
        self.num_workers = config.get("num_workers", 0)
        self.pin_memory = config.get("pin_memory", False)

        # Early stopping
        self.patience = config.get("early_stopping_patience", 10)

        # Model save path
        self.model_dir = Path(config.get("model_dir", "./outputs/models"))

    def train(
        self,
        model: nn.Module,
        train_dataset: NBASequenceDataset,
        val_dataset: NBASequenceDataset,
        target_stat: str = "points",
        model_name: str = "model",
    ) -> dict:
        """Run the full training loop.

        Args:
            model: PyTorch model to train.
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
            target_stat: Which stat to train on.
            model_name: Name for saving checkpoints.

        Returns:
            Dict with training history (train_loss, val_loss per epoch).
        """
        model = model.to(self.device)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=NBASequenceDataset.collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=NBASequenceDataset.collate_fn,
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )

        # Callbacks
        checkpoint_path = self.model_dir / f"{model_name}_best.pt"
        callbacks = CallbackList(
            [
                EarlyStopping(patience=self.patience),
                ModelCheckpoint(str(checkpoint_path)),
            ]
        )

        history = {"train_loss": [], "val_loss": []}

        logger.info(
            "Starting training: %d epochs, batch_size=%d, device=%s",
            self.epochs,
            self.batch_size,
            self.device,
        )

        for epoch in range(1, self.epochs + 1):
            # Training phase
            model.train()
            train_losses = []

            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()

                distributions = model(
                    sequence_features=batch["sequence_features"],
                    static_features=batch["static_features"],
                    mask=batch["mask"],
                )

                # Negative log-likelihood loss
                dist = distributions[target_stat]
                targets = batch["target"].float()
                loss = -dist.log_prob(targets).mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.gradient_clip_norm
                )
                optimizer.step()

                train_losses.append(loss.item())

            # Validation phase
            model.eval()
            val_losses = []

            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    distributions = model(
                        sequence_features=batch["sequence_features"],
                        static_features=batch["static_features"],
                        mask=batch["mask"],
                    )
                    dist = distributions[target_stat]
                    targets = batch["target"].float()
                    loss = -dist.log_prob(targets).mean()
                    val_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            scheduler.step(val_loss)

            logger.info(
                "Epoch %d/%d - train_loss: %.4f, val_loss: %.4f, lr: %.6f",
                epoch,
                self.epochs,
                train_loss,
                val_loss,
                optimizer.param_groups[0]["lr"],
            )

            # Callbacks
            should_stop = callbacks.on_epoch_end(epoch, val_loss, model)
            if should_stop:
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

        # Load best model
        if checkpoint_path.exists():
            model.load_state_dict(
                torch.load(checkpoint_path, map_location=self.device, weights_only=True)["state_dict"]
            )
            logger.info("Loaded best model from %s", checkpoint_path)

        return history
