from config import CFG

import math
from collections import OrderedDict


import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
from torchmetrics.classification import MultilabelAveragePrecision as mAP

from sklearn.metrics import average_precision_score as APS

from mamba_ssm import Mamba




class LMmamba(L.LightningModule):
    def __init__(self, batch, input_dim, input_dim_embedding, hidden_dim, num_layers, dropout, out_dim, learning_rate, weight_decay):
        super().__init__()
        self.save_hyperparameters()
        self.val_preds = []
        self.val_y = []
        self.metrics_map = mAP(3,thresholds=None, average='micro')
        self.metrics_ap = mAP(3,thresholds=None, average='none')


        self.embedding = nn.Embedding(num_embeddings=input_dim_embedding, embedding_dim=hidden_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=0)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.mamba_blocks = nn.Sequential(OrderedDict([(f'mamba_{i}', MambaSE(hidden_dim, dropout)) for i in range(num_layers)]))
        self.mlp = GatedMLP(hidden_dim, out_features=out_dim,)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0,2,1)
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = x.permute(0,2,1)
        x = self.mamba_blocks(x)
        x = x.mean(dim=1)
        x = self.mlp(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = self.process_batch(batch)
        logits = self(x)
        preds = torch.sigmoid(logits)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = self.process_batch(batch)
        logits = self(x)
        preds = torch.sigmoid(logits)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.val_preds.append(preds)
        self.val_y.append(y)
        return loss

    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_preds, 0)
        y = torch.cat(self.val_y, 0).to(torch.int)
        APs = self.metrics_ap(preds, y)
        for i in range(3):
            self.log(f'val_AP_bind{i}',APs[i], sync_dist=True)
        meanAP = self.metrics_map(preds, y)
        self.log('val_mAP', meanAP, prog_bar=True, sync_dist=True)
        self.val_preds.clear()
        self.val_y.clear()


    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        self.log('test_loss', loss)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        return optimizer

    def process_batch(self, batch):
        X, y = batch[0].clone().long(), batch[1].clone()
        return X, y

class MambaSE(nn.Module):
    def __init__(self,hidden_dim, dropout):
        super().__init__()
        self.dropout = dropout
        self.mamba = Mamba(
            d_model=hidden_dim,
            d_state=16,
            d_conv=3,
            expand=2
            )
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.se = SELayer(hidden_dim)
    
    def forward(self, x):
        x = self.mamba(x)
        x = x.permute(0,2,1)
        x = self.norm(x)
        x = self.se(x)
        x = x.permute(0,2,1)
        x = F.dropout(x, self.dropout)
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class GatedMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        activation=F.silu,
        bias=False,
        multiple_of=128,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = (
            hidden_features if hidden_features is not None else int(8 * in_features / 3)
        )
        hidden_features = (hidden_features + multiple_of - 1) // multiple_of * multiple_of
        self.fc1 = nn.Linear(in_features, 2 * hidden_features, bias=bias, **factory_kwargs)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, **factory_kwargs)

    def forward(self, x):
        y = self.fc1(x)
        y, gate = y.chunk(2, dim=-1)
        y = y * self.activation(gate)
        y = self.fc2(y)
        return y



class DemoModel(L.LightningModule):
    def __init__(self, input_dim=142, input_dim_embedding=37, hidden_dim=128, num_filters=32, output_dim=3, lr=1e-3, weight_decay=1e-6):
        super().__init__()
        self.save_hyperparameters()

        self.embedding = nn.Embedding(num_embeddings=self.hparams.input_dim_embedding, embedding_dim=self.hparams.hidden_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(in_channels=self.hparams.hidden_dim, out_channels=self.hparams.num_filters, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv1d(in_channels=self.hparams.num_filters, out_channels=self.hparams.num_filters*2, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv1d(in_channels=self.hparams.num_filters*2, out_channels=self.hparams.num_filters*3, kernel_size=3, stride=1, padding=0)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(self.hparams.num_filters*3, 1024)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.output = nn.Linear(512, self.hparams.output_dim)

    def forward(self, x):
        x = self.embedding(x).permute(0,2,1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.global_max_pool(x).squeeze(2)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.output(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        self.log('val_loss', loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer

    def process_batch(self, batch):
        X, y = batch
        X, y = X.clone(), y.clone()
        return X, y
