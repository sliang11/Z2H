# Modified by Shen Liang and Cihang Yu

import os
import random
import pickle
import numpy as np

import torch
import torch.nn as nn
import pytorch_lightning as pl
# from pytorch_lightning import loggers as pl_loggers
from functools import partial
from sklearn.metrics import precision_score, recall_score, f1_score
from util.pytorch_util import set_seed
from util.file_util import DirProcessor
from util.seeg_file_util import load_seanet_format
from deep_learning.EEGPT.downstream.adapter import EEGSpatialAdapter
import argparse

# EEGPT
from deep_learning.EEGPT.downstream.Modules.models.EEGPT_mcae import EEGTransformer
from deep_learning.EEGPT.downstream.Modules.Network.utils import Conv1dWithConstraint, LinearWithConstraint


from pathlib import Path

CKPT_PATH = "eegpt_mcae_58chs_4s_large4E.ckpt"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT_PATH = os.path.join(SCRIPT_DIR, CKPT_PATH)



class NumpyEEGDataset(torch.utils.data.Dataset):
    def __init__(self, examples: np.ndarray, labels: np.ndarray):
        self.X = examples.astype(np.float32, copy=False)
        self.y = labels.astype(np.int64, copy=False)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()  # (C,T)
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return {"inputs": x, "labels": y, "sample_id": torch.tensor(idx, dtype=torch.long)}


class LitEEGPTCausal(pl.LightningModule):
    def __init__(
        self,
        ckpt_path: str,
        input_channels: int,
        target_channels: int = 19,
        target_t: int = 512,
        lr: float = 4e-4,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay

        self.target_encoder = EEGTransformer(
            img_size=[target_channels, target_t],
            patch_size=32 * 2,
            patch_stride=32,
            embed_num=4,
            embed_dim=512,
            depth=8,
            num_heads=8,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

        use_channels_names = [
            'FP1', 'FP2',
            'F7', 'F3', 'FZ', 'F4', 'F8',
            'T7', 'C3', 'CZ', 'C4', 'T8',
            'P7', 'P3', 'PZ', 'P4', 'P8',
            'O1', 'O2'
        ]

        chans_id = self.target_encoder.prepare_chan_ids(use_channels_names).long()
        if (chans_id < 0).any():
            chans_id = chans_id[chans_id >= 0]
        self.chans_id = chans_id

        pretrain_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "state_dict" not in pretrain_ckpt:
            raise KeyError("Le checkpoint ne contient pas 'state_dict'.")

        target_encoder_state = {}
        for k, v in pretrain_ckpt["state_dict"].items():
            if k.startswith("target_encoder."):
                target_encoder_state[k[len("target_encoder."):]] = v
        self.target_encoder.load_state_dict(target_encoder_state, strict=True)

        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.target_encoder.eval()

        self.adapter = EEGSpatialAdapter(
            input_channels=input_channels,
            target_channels=target_channels,
            input_length=None,
            target_length=target_t,
        )

        self.chan_conv = Conv1dWithConstraint(target_channels, target_channels, 1, max_norm=1)

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, target_t)
            dummy = dummy - dummy.mean(dim=-2, keepdim=True)
            dummy = self.adapter(dummy)
            dummy = self.chan_conv(dummy)
            z = self.target_encoder(dummy, self.chans_id.to(dummy))
            K = int(z.shape[1])

        self.K_tokens = K
        print("[INFO] Nombre de tokens de canaux K =", K, flush=True)

        self.drop = nn.Dropout(p=0.50)
        self.linear_probe1 = LinearWithConstraint(2048, 16, max_norm=1)
        self.linear_probe2 = LinearWithConstraint(K * 16, 2, max_norm=0.25)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x - x.mean(dim=-2, keepdim=True)
        x = self.adapter(x)
        x = self.chan_conv(x)
        with torch.no_grad():
            z = self.target_encoder(x, self.chans_id.to(x))
        h = z.flatten(2)
        h = self.linear_probe1(self.drop(h))
        h = h.flatten(1)
        logits = self.linear_probe2(h)
        return logits

    def _step(self, batch, stage: str):
        x = batch["inputs"]
        y = batch["labels"].long()
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == y).float().mean()
        self.log(f"{stage}_loss", loss, on_epoch=True, on_step=False)
        self.log(f"{stage}_acc", acc, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "valid")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def configure_optimizers(self):
        params = (
            list(self.adapter.parameters()) +
            list(self.chan_conv.parameters()) +
            list(self.linear_probe1.parameters()) +
            list(self.linear_probe2.parameters())
        )
        return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)


@torch.no_grad()
def predict_labels(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device):
    model.eval()
    all_preds, all_true = [], []
    for batch in loader:
        x = batch["inputs"].to(device)
        y = batch["labels"].cpu().numpy()
        logits = model(x)
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.append(preds)
        all_true.append(y)
    y_pred = np.concatenate(all_preds).astype(np.int64)
    y_true = np.concatenate(all_true).astype(np.int64)
    return y_true, y_pred


def save_required_outputs(save_path: str, y_pred: np.ndarray, prf_tuple: tuple, model: pl.LightningModule):
    os.makedirs(save_path, exist_ok=True)

    raw_pred_path = os.path.join(save_path, "raw_predictions.pkl")
    with open(raw_pred_path, "wb") as f:
        pickle.dump(y_pred.astype(np.int64), f)

    prf_path = os.path.join(save_path, "prediction_prf_with_beta=1.pkl")
    with open(prf_path, "wb") as f:
        pickle.dump(tuple(prf_tuple), f)

    model_path = os.path.join(save_path, "model_state_dict.pt")
    torch.save({"state_dict": model.state_dict(), "hparams": getattr(model, "hparams", None)}, model_path)

    print("[INFO] Saved:", raw_pred_path, flush=True)
    print("[INFO] Saved:", prf_path, flush=True)
    print("[INFO] Saved:", model_path, flush=True)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("data_id", type=str)
    parser.add_argument("--channel", type=int, default=None)
    parser.add_argument('--rand_seed', type=int, default=0)
    parser.add_argument('--main_dl_data_path', type=str, default=None)
    parser.add_argument('--main_dl_result_path', type=str, default=None)

    # Checkpoint
    parser.add_argument("--ckpt_path", type=str, default=DEFAULT_CKPT_PATH)

    # Training hyperparams
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--target_t", type=int, default=512)
    parser.add_argument("--target_channels", type=int, default=19)

    args = parser.parse_args()

    channel_id = args.channel if args.channel is not None else 'all_ch'

    main_dl_data_path = args.main_dl_data_path
    if main_dl_data_path is None:
        main_dl_data_path = os.path.join(str(Path(__file__).resolve().parent.parent.parent.parent),
                                         "data", 'Z2H_dl_data')
    main_dl_result_path = args.main_dl_result_path
    if main_dl_result_path is None:
        main_dl_result_path = os.path.join(str(Path(__file__).resolve().parent.parent.parent.parent),
                                           "dl_result")

    dl_result_path = os.path.join(main_dl_result_path, 'EEGPT', f'{args.data_id}_{channel_id}')
    DirProcessor.create_dir(dl_result_path)

    set_seed(args.rand_seed)

    all_X, all_y = [], []
    for tvt_id in ('train', 'val', 'test'):
        p_ex, n_ex = load_seanet_format(tvt_id, f'{args.data_id}_{channel_id}', main_dl_data_path)
        X = np.concatenate((p_ex, n_ex))
        y = np.concatenate((np.ones(len(p_ex)), np.zeros(len(n_ex))))
        all_X.append(X)
        all_y.append(y)
    X_train, X_val, X_test = all_X
    y_train, y_val, y_test = all_y

    train_ds = NumpyEEGDataset(X_train, y_train)
    val_ds   = NumpyEEGDataset(X_val, y_val)
    test_ds  = NumpyEEGDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    input_channels = int(X_train.shape[1])
    print("[INFO] input_channels =", input_channels,
          "| target_channels =", args.target_channels,
          "| target_t =", args.target_t, flush=True)

    model = LitEEGPTCausal(
        ckpt_path=args.ckpt_path,
        input_channels=input_channels,
        target_channels=args.target_channels,
        target_t=args.target_t,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    trainer = pl.Trainer(
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=args.max_epochs,
        enable_checkpointing=False,
        # logger=[
        #     pl_loggers.TensorBoardLogger("./logs/", name="eegpt_server_tb"),
        #     pl_loggers.CSVLogger("./logs/", name="eegpt_server_csv"),
        # ],
        logger=False,
        log_every_n_steps=1000_000,
        enable_progress_bar=False,
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    y_true, y_pred = predict_labels(model, test_loader, device)

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall    = float(recall_score(y_true, y_pred, zero_division=0))
    f1        = float(f1_score(y_true, y_pred, zero_division=0))
    prf_tuple = (precision, recall, f1)

    print("[INFO] save_path =", dl_result_path, flush=True)

    save_required_outputs(dl_result_path, y_pred, prf_tuple, model)
    print("[INFO] Done. PRF =", prf_tuple, flush=True)


if __name__ == "__main__":
    main()
