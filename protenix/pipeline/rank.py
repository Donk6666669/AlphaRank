import logging
from typing import Any, Dict

from hydra.utils import instantiate
from lightning import LightningModule

from protenix.utils.torchtool import get_model_size_mb

logger = logging.getLogger(__name__)


class RankLitModule(LightningModule):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(cfg)

        # initialize the model from configuration
        self.model = instantiate(self.hparams.model)

        # initialize the criterion from configuration
        self.criterion = instantiate(self.hparams.criterion)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your
        optimization.

        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = instantiate(self.hparams.optim, self.parameters())
        scheduler = instantiate(
            self._set_num_training_steps(self.hparams.scheduler), optimizer
        )
        # torch's scheduler is epoch-based, but transformers' is step-based
        interval = (
            "step"
            if self.hparams.scheduler._target_.startswith("transformers")
            else "epoch"
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": interval,
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    def _set_num_training_steps(self, scheduler_cfg):
        if (
            "num_training_steps" in scheduler_cfg
            and scheduler_cfg["num_training_steps"] == "auto"
        ):
            scheduler_cfg = dict(scheduler_cfg)
            logger.info("Computing number of training steps...")
            scheduler_cfg["num_training_steps"] = (
                self.trainer.estimated_stepping_batches
            )

            if self.global_rank == 0:
                logger.info(
                    f"Training steps: {scheduler_cfg['num_training_steps']}"
                )
        return scheduler_cfg

    def on_train_start(self):

        self.log(
            "model_size/total",
            get_model_size_mb(self.model),
            rank_zero_only=True,
            logger=True,
            sync_dist=True,
        )

    def step(self, batch: Any):
        criterion_inputs = {
            "label": batch.pop("label"),
            "hard": batch.pop("hard"),
            "idx": batch.pop("idx"),
        }
        criterion_inputs.update(self.model(**batch))
        outputs = self.criterion(criterion_inputs)
        return outputs

    def on_train_epoch_start(self):
        self.criterion.reset()

    def training_step(self, batch: Any, batch_idx: int):
        outputs = self.step(batch)

        log_interval = 10
        if self.global_step % log_interval == 0:
            for key, value in outputs.items():
                self.log(
                    f"train/{key}",
                    value,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                    sync_dist=True,
                )
        return outputs["loss"]

    def on_validation_epoch_start(self):
        # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#lightning-hooks
        metrics = self.criterion.compute(verbose=True)
        for name, value in metrics.items():
            self.log(
                f"train/{name}",
                value,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        self.criterion.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        outputs = self.step(batch)
        return outputs

    def on_validation_epoch_end(self):
        metrics = self.criterion.compute(
            verbose=True,
            save_name=f"val_{self.current_epoch}_{self.global_rank}.json",
        )
        for name, value in metrics.items():
            self.log(
                f"val/{name}",
                value,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        self.criterion.reset()

    def test_step(self, batch: Any, batch_idx: int):
        outputs = self.step(batch)
        return outputs

    def on_test_epoch_end(self):
        metrics = self.criterion.compute(
            verbose=True,
            save_name=f"test_{self.current_epoch}_{self.global_rank}.json",
        )
        for name, value in metrics.items():
            self.log(
                f"test/{name}",
                value,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        self.criterion.reset()

    def predict_step(self, batch):
        self.model.eval()
        return self.model(batch)["pred"]
