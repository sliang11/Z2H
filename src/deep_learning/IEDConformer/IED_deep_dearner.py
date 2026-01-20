import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from util.seeg_file_util import load_seanet_format
from util.pytorch_util import set_seed
from util.evaluation_util import prf
from util.file_util import DirProcessor, FileWriter
from torchmetrics.classification import F1Score
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from copy import deepcopy
import os


class IEDDeepNet(nn.Module):

    def __init__(self, n_channels, ts_len, n_classes=2):
        super().__init__()

        self.n_channels = n_channels
        self.ts_len = ts_len
        self.n_classes = n_classes

        # for k, v in kwargs.items():
        #     setattr(self, k, v)

    def forward(self, x: torch.Tensor):  # x.shape == (batch_size, n_channels, ts_len)
        assert x.ndim == 3
        assert x.shape[1] == self.n_channels and x.shape[2] == self.ts_len
        logits = self._core(x)

        logit_dim = self.n_classes if self.n_classes > 2 else 1
        assert logits.shape == (x.shape[0], logit_dim)
        return logits

    def _core(self, x: torch.Tensor):   # implement this method
        raise NotImplementedError()


# define the LightningModule
class LitIED(L.LightningModule):
    def __init__(self, n_channels, ts_len,
                 ied_net_cls,
                 optimizer_cls, optimizer_kwargs: dict,
                 scheduler_cls=None,
                 scheduler_kwargs: dict = {},
                 scheduler_interval: str = "epoch",  # "epoch" | "step"
                 scheduler_monitor: str | None = 'val_loss',  # only for ReduceLROnPlateau
                 criterion_cls=None,
                 criterion_kwargs: dict = {},
                 pretrain_kwargs: dict = {},
                 n_classes=2,
                 ):
        super().__init__()

        assert issubclass(ied_net_cls, IEDDeepNet)
        assert issubclass(optimizer_cls, torch.optim.Optimizer)
        if scheduler_cls is not None:
            assert issubclass(
                scheduler_cls,
                (torch.optim.lr_scheduler.LRScheduler,
                 torch.optim.lr_scheduler.ReduceLROnPlateau)
            )

        if criterion_cls is None:
            if n_classes == 2:
                self.criterion = torch.nn.BCEWithLogitsLoss(**criterion_kwargs)
            else:
                raise NotImplementedError()
        else:
            assert issubclass(criterion_cls, nn.Module)
            self.criterion = criterion_cls(**criterion_kwargs)

        self.ied_net = ied_net_cls(n_channels, ts_len, n_classes=n_classes)

        # optimizer config
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs

        # scheduler config
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.scheduler_interval = scheduler_interval
        self.scheduler_monitor = scheduler_monitor

        # load pretrained model (and potentially optimizer, scheduler, random state, as necessary)
        # if pretrain_kwargs is not None:
        self.load_pretrain(**pretrain_kwargs)

        # validation config
        if n_classes == 2:
            self.val_f1 = F1Score(task='binary', threshold=0.5, sync_on_compute=True)
        else:
            raise NotImplementedError()

        self.save_hyperparameters(ignore=["ied_net"])  # ignore the model as it is too big

    def load_pretrain(self, **kwargs):  # Cihang
        pass  # Implement this method as needed (only for foundation models)

        # Possible things to load here: 1) model parameters; 2) optimizer state (optional); 3) scheduler state (optional); 4) random state (optional)

    def forward(self, x: torch.Tensor):
        return self.ied_net(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        loss = self._calc_loss_one_batch(batch)

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    # --------------------------------------------------
    # validation (needed for ReduceLROnPlateau)
    # --------------------------------------------------
    def validation_step(self, batch, batch_idx):
        loss = self._calc_loss_one_batch(batch, is_val=True)
        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def on_validation_epoch_end(self):
        f1 = self.val_f1.compute()

        self.log(
            "val_f1",
            f1,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.val_f1.reset()

    def _calc_loss_one_batch(self, batch, is_val: bool = False):    # is_val: indicates whether this is the validation step; if so, update val_f1
        x, y = batch

        assert x.ndim == 3
        assert y.ndim == 1
        assert x.shape[0] == y.shape[0]

        # y should be among 0 ~ n_classes-1
        assert torch.all(y >= 0)
        assert torch.all(y < self.ied_net.n_classes)

        out_logits = self(x)

        if self.ied_net.n_classes > 2:
            target_logits = F.one_hot(y, self.ied_net.n_classes).float()
            if is_val:
                raise NotImplementedError()
        else:
            target_logits = y.unsqueeze(1).float()
            if is_val:
                self.val_f1.update((torch.sigmoid(out_logits) >= 0.5).squeeze(1).long(), y.long())

        loss = self.criterion(out_logits, target_logits)

        return loss

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        return self(x)

    # --------------------------------------------------
    # optimizer & scheduler
    # --------------------------------------------------
    def configure_optimizers(self):
        optimizer = self.optimizer_cls(
            self.parameters(),
            **self.optimizer_kwargs,
        )

        # ---- no scheduler ----
        if self.scheduler_cls is None:
            return optimizer

        scheduler = self.scheduler_cls(
            optimizer,
            **self.scheduler_kwargs,
        )

        # ---- ReduceLROnPlateau (special case) ----
        if issubclass(self.scheduler_cls, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if self.scheduler_monitor is None:
                raise ValueError(
                    "ReduceLROnPlateau requires `scheduler_monitor` "
                    "(e.g. 'val_loss')."
                )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.scheduler_monitor,
                    "interval": "epoch",
                },
            }

        # ---- all other schedulers ----
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.scheduler_interval,
            },
        }


class IEDDeepLearner(object):

    def __init__(self, dl_result_save_path,

                 base_seanet_save_path,
                 main_dl_data_path,
                 rand_seed,

                 batch_size, n_epochs,
                 ied_net_cls,
                 optimizer_cls, optimizer_kwargs: dict,
                 scheduler_cls=None,
                 scheduler_kwargs: dict = {},
                 scheduler_monitor: str | None = 'val_loss',  # only for ReduceLROnPlateau
                 criterion_cls=None,
                 criterion_kwargs: dict = {},
                 pretrain_kwargs: dict = {},

                 trainer_max_time='00:15:00:00',
                 ):

        self.dl_result_save_path = dl_result_save_path
        DirProcessor.create_dir(dl_result_save_path)

        self.rand_seed = rand_seed
        set_seed(self.rand_seed)

        self.dataset_by_tvt_id = {}
        for tvt_id in ('train', 'valid', 'test'):
            positive_examples, negative_examples = load_seanet_format(
                tvt_id, base_seanet_save_path, main_dl_data_path)
            examples = np.concatenate((positive_examples, negative_examples))
            labels = np.concatenate((np.ones(len(positive_examples)), np.zeros(len(negative_examples)))).astype(np.int64)
            if tvt_id == 'test':
                self.test_real_labels = deepcopy(labels)

            x, y = torch.from_numpy(examples).float(), torch.from_numpy(labels).long()
            self.n_channels, self.ts_len = x.shape[1], x.shape[2]
            self.dataset_by_tvt_id[tvt_id] = TensorDataset(x.clone(), y.clone())


        self.batch_size = batch_size
        self.loader_by_tvt_id = {
            tvt_id: DataLoader(dataset, batch_size=self.batch_size, shuffle=(tvt_id == 'train'))
            for (tvt_id, dataset) in self.dataset_by_tvt_id.items()
        }

        self.lit_ied = LitIED(
            self.n_channels, self.ts_len,
            ied_net_cls,
            optimizer_cls, optimizer_kwargs,
            scheduler_cls=scheduler_cls,
            scheduler_kwargs=scheduler_kwargs,
            scheduler_monitor=scheduler_monitor,  # only for ReduceLROnPlateau
            criterion_cls=criterion_cls,
            criterion_kwargs=criterion_kwargs,
            pretrain_kwargs=pretrain_kwargs
        )

        self.logger = TensorBoardLogger(
            save_dir=f'{dl_result_save_path}/logs',
            name="",
            version="",
        )

        self.checkpoint_cb = ModelCheckpoint(
            dirpath=f'{dl_result_save_path}/checkpoints',
            filename="epoch{epoch:03d}-valF1{val_f1:.4f}",
            monitor="val_f1",
            mode="max",
            save_top_k=1,
            save_last=True,
        )

        self.n_epochs = n_epochs
        self.trainer = L.Trainer(
            max_epochs=self.n_epochs, max_time=trainer_max_time,
            accelerator='gpu', devices=1,
            logger=self.logger,
            callbacks=[self.checkpoint_cb],
            enable_progress_bar=False,
        )

    def train(self):
        self.trainer.fit(
            model=self.lit_ied,
            train_dataloaders=self.loader_by_tvt_id['train'],
            val_dataloaders=self.loader_by_tvt_id['valid'],
            ckpt_path="last",
        )

        self.best_val_f1 = float(self.checkpoint_cb.best_model_score)
        save_fname = os.path.join(self.dl_result_save_path, f'best_val_f1.pkl')
        FileWriter.dump_pickle(self.best_val_f1, save_fname)

    def test(self):
        all_out_logits = self.trainer.predict(
            model=self.lit_ied,
            dataloaders=self.loader_by_tvt_id['test'],
            ckpt_path='best'
        )

        all_out_logits = torch.cat(all_out_logits, dim=0)
        predicted_labels = (torch.sigmoid(all_out_logits) >= 0.5).long().squeeze(1).cpu().numpy()
        p, r, f = prf(self.test_real_labels, predicted_labels)

        print(f'Testing complete, with p = {p}, r = {r}, f = {f}')

        raw_pred_save_fname = os.path.join(self.dl_result_save_path, f'raw_predictions.pkl')
        FileWriter.dump_pickle(predicted_labels, raw_pred_save_fname)

        prf_save_fname = os.path.join(self.dl_result_save_path, f'prediction_prf_with_beta=1.pkl')
        FileWriter.dump_pickle((p, r, f), prf_save_fname)
