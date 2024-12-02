# set some constants
from setup import *

import hydra
import torch
import torch.nn.functional as F
import lightning as L

from myutils import MetricLogger
from dataset import cover_dataloader
from models_mae import MaskedAutoencoderViT


class MAEModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = MaskedAutoencoderViT(
            img_size=(84, 52),
            patch_size=(4, 4),
            in_chans=1,
            embed_dim=768,
            depth=16,
            decoder_depth=6,
            decoder_embed_dim=384,
            mlp_ratio=4,
            num_heads=12,
            decoder_num_heads=6,
        )
        self.ema = MetricLogger(alpha=0.99)

    def _step(self, batch, kind):
        x = batch["anchor_tr"]
        x = F.pad(x, (0, 2, 0, 0), "constant", 0).unsqueeze(1)
        x = (x - MEAN) / STD

        loss, pred, mask = self.model(x, mask_ratio=0.75)

        self._log_metrics(loss, kind)
        return loss        

    def _log_metrics(self, loss, kind):
        if kind == "train":
            self.ema.update(loss.item())
            loss = self.ema.exp
        metrics = {
            f"{kind}_loss": loss,
            "lr": self.optimizers().optimizer.param_groups[0]["lr"],
        }
        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=kind == "train",
            on_epoch=True,
        )

    def training_step(self, batch):
        return self._step(batch, "train")

    def validation_step(self, batch):
        return self._step(batch, "valid")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.epochs * len(self.train_dataloader()),
            eta_min=self.cfg.learning_rate * 0.1,
        )

        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return cover_dataloader(data_split="train", **self.cfg)

    def val_dataloader(self):
        return cover_dataloader(data_split="val", **self.cfg)


@hydra.main(version_base=None, config_path="config", config_name="mae")
def main(cfg):
    if cfg.checkpoint_path:
        model = MAEModule.load_from_checkpoint(cfg.checkpoint_path)
    else:
        model = MAEModule(cfg)

    callbacks = [
        L.pytorch.callbacks.TQDMProgressBar(leave=True),
        L.pytorch.callbacks.ModelCheckpoint(
            filename="best",
            monitor="valid_loss",
            mode="min",
            save_last=True,
            every_n_epochs=5,
        ),
    ]

    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        precision="bf16-mixed",
        default_root_dir="mae_logs",
    )

    trainer.fit(model)

    torch.save(model.model.state_dict(), "mae.pt")


if __name__ == "__main__":
    main()