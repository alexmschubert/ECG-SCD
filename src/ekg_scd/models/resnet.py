from __future__ import division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from ekg_scd.models.cumulative_probability_layer import Cumulative_Probability_Layer

from . import base

class EKGResNetModel(base.Model):
    def __init__(self, **kwargs):
        super(EKGResNetModel, self).__init__(**kwargs)
        self.net = EKGResNet(**kwargs)
        self.covariate_conditioning = kwargs.get("covariate_conditioning", None)
        self.regress = np.array(kwargs.get("regress"))
        self.two_outs = kwargs.get("two_outs", False)

        self.classify_lossfun = base.NanBCEWithLogitsLoss(reduction="mean")
        self.reg_lossfun = nn.MSELoss(reduction="mean")

        self.n_outputs = kwargs.get("n_outputs")
        last_dim = self.net.num_last_active

        self.out = nn.Linear(last_dim, self.n_outputs)
        if self.two_outs:
            self.out = nn.Sequential(
                nn.Linear(last_dim, 128),
                nn.ReLU(),
                nn.Linear(128, self.n_outputs)
            )

        if self.covariate_conditioning is not None:
            # Define the neural network block
            total_in_features = last_dim + len(self.covariate_conditioning)
            self.out = nn.Sequential(
                nn.Linear(total_in_features, 128),  # Linear layer with input dimension of total_in_features and output dimension of 128
                nn.ReLU(),                          # ReLU activation function
                nn.Linear(128, self.n_outputs)        # Linear layer with input dimension of 128 and output dimension of out_features
            )


    def lossfun(self, data, target, x_cond=None):
    
        logit = self.forward(data, x_cond)
        if np.min(self.regress) == 1:
            pred_loss = self.reg_lossfun(logit, target)
        elif np.max(self.regress) == 0:
            pred_loss = self.classify_lossfun(logit, target)
        else:
            regression_loss = self.reg_lossfun(logit[:, self.regress], target[:, self.regress])
            classify_loss = self.classify_lossfun(logit[:, ~self.regress], target[:, ~self.regress])
            pred_loss = regression_loss * 10 + classify_loss
        return pred_loss, logit

    def forward(self, data, x_cond=None):
        deep = self.net(data)
        if self.covariate_conditioning is not None:
            deep = torch.cat([deep,x_cond], dim=1)
        deep = self.out(deep)
        return deep

    def fit(self, train_loader, val_loader, **kwargs):
        self.fit_res = base.fit_model(self, train_loader, val_loader, **kwargs)
        return self.fit_res


class EKGResNetAuxModel(base.Model):
    def __init__(self, **kwargs):
        super(EKGResNetAuxModel, self).__init__(**kwargs)
        self.net = EKGResNet(**kwargs)
        self.auxiliary_layers = kwargs.get("auxiliary_layers", None)
        self.covariate_conditioning = kwargs.get("covariate_conditioning", None)
        self.regress = np.array(kwargs.get("regress"))
        self.two_outs = kwargs.get("two_outs", False)

        self.classify_lossfun = base.NanBCEWithLogitsLoss(reduction="mean")
        self.reg_lossfun = nn.MSELoss(reduction="mean")

        self.n_outputs = kwargs.get("n_outputs")
        last_dim = self.net.num_last_active
        self.out = nn.Linear(last_dim, self.n_outputs)

        if self.covariate_conditioning is not None:
            print('conditioning covars')
            # Define the neural network block
            total_in_features = last_dim + len(self.covariate_conditioning)
            self.out = nn.Sequential(
                nn.Linear(total_in_features, 128),  # Linear layer with input dimension of total_in_features and output dimension of 128
                nn.ReLU(),                          # ReLU activation function
                nn.Linear(128, self.n_outputs)        # Linear layer with input dimension of 128 and output dimension of out_features
            )

        # Replace the model's final layer with a new set of layers
        in_features = self.out[-1].in_features
        layers = list(self.out.children())[:-1]
        for num in self.auxiliary_layers:
            layers.append(nn.Linear(in_features, num))
            layers.append(nn.ReLU())
            in_features = num
        layers.append(nn.Linear(in_features, self.n_outputs))
        self.out = nn.Sequential(*layers)


    def lossfun(self, data, target, x_cond=None):
    
        logit = self.forward(data, x_cond)
        if np.min(self.regress) == 1:
            pred_loss = self.reg_lossfun(logit, target)
        elif np.max(self.regress) == 0:
            pred_loss = self.classify_lossfun(logit, target)
        else:
            regression_loss = self.reg_lossfun(logit[:, self.regress], target[:, self.regress])
            classify_loss = self.classify_lossfun(logit[:, ~self.regress], target[:, ~self.regress])
            pred_loss = regression_loss * 10 + classify_loss
        return pred_loss, logit

    def forward(self, data, x_cond=None):
        deep = self.net(data)
        if self.covariate_conditioning is not None:
            deep = torch.cat([deep,x_cond], dim=1)
        deep = self.out(deep)
        return deep

    def fit(self, train_loader, val_loader, **kwargs):
        self.fit_res = base.fit_model(self, train_loader, val_loader, **kwargs)
        return self.fit_res


class EKGDeepWideResNetModel(EKGResNetModel):
    """wide and deep EKG resnet.  Expects the last `dim_wide` dimensions
    to be linearly added into the final prediction."""

    def __init__(self, **kwargs):
        super(EKGDeepWideResNetModel, self).__init__(**kwargs)
        self.dim_wide = kwargs.get("dim_wide")
        dropout = kwargs.get("dropout", 0.4)
        last_dim = self.net.num_last_active + self.dim_wide
        self.out = nn.Sequential(
            nn.Linear(last_dim, int(0.5 * last_dim)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(0.5 * last_dim), int(0.25 * last_dim)),
            nn.ReLU(),
            nn.Linear(int(0.25 * last_dim), self.n_outputs),
        )

    def forward(self, data):
        """this module takes in a batch_sz x (C x T + dim_wide)"""
        wide = data[:, 0, -self.dim_wide :]
        data = data[:, :, : -self.dim_wide]

        # wide + EKG representation
        deep = self.net(data)
        zout = torch.cat([deep, wide], 1)
        return self.out(zout)


class EKGResNet(base.Model):
    """Residual network
    
    Args:
      - n_channels: number of leads used (batches are size bsz x n_channels x n_samples)
      - n_samples : how long each tracing is
      - n_outputs : number of features to predict
      - num_rep_blocks: how deep the network is --- how many repeated internal
          resnet blocks
    """

    def __init__(self, **kwargs):
        super(EKGResNet, self).__init__(**kwargs)
        n_channels = kwargs.get("n_channels")
        n_samples = kwargs.get("n_samples")
        n_outputs = kwargs.get("n_outputs")
        num_rep_blocks = kwargs.get("num_rep_blocks", 16)  # depth of the network #16 base
        kernel_size = kwargs.get("kernel_size", 16)
        conv_channels = kwargs.get("conv_channels", 128)
        self.verbose = kwargs.get("verbose", False)
        dropout = kwargs.get("dropout", 0.5)

        self.cumulative_predictor = kwargs.get("cumulative_predictor", False)
        self.output_names = kwargs.get("output_names", None)
        self.rep_mp = kwargs.get("rep_mp", 4)

        # store dimension info
        self.n_channels, self.n_samples, self.n_outputs = n_channels, n_samples, n_outputs

        # track signal length so you can appropriately set the fully connected layer
        self.seq_len = n_samples

        # set up first block
        self.block0 = nn.Sequential(
            nn.Conv1d(in_channels=n_channels, out_channels=conv_channels, kernel_size=kernel_size, padding=(kernel_size // 2)),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
        )
        self.seq_len += 1

        # input --- output is bsz x 64 x ceil(n_samples/2)
        self.block1 = BlockOne(in_channels=conv_channels, out_channels=conv_channels, kernel_size=kernel_size, dropout=dropout)
        self.seq_len = self.seq_len // 2

        # repeated block
        self.num_rep_blocks = num_rep_blocks
        self.num_layers = 3 + 2 * num_rep_blocks + 1  # each rep block is 2
        in_features = conv_channels
        num_max_pool = 0

        self.rep_blocks = nn.ModuleList()
        for l in range(num_rep_blocks):
            # determine number of output features
            out_features = in_features
            b = RepBlock(in_channels=in_features, out_channels=out_features, kernel_size=16, dropout=dropout)
            self.rep_blocks.append(b)

            # count how many features are removed
            if l % self.rep_mp == 0:
                num_max_pool += 1

        # update seq_len
        self.seq_len = self.seq_len // (2**num_max_pool)
        print("\nSEQ_LEN: ", self.seq_len, "\nOUT_FEATURES: ", out_features, "\n")
        self.num_last_active = int(self.seq_len * out_features)

        # output block
        self.block_out = nn.Sequential(nn.BatchNorm1d(in_features), nn.ReLU())

        print("net thinks it will have")
        print("  seq_len     :", self.seq_len)
        print("  last active :", self.num_last_active)

        # maxpool, used throughout
        self.mp = nn.MaxPool1d(2, stride=2)

        # Add cumulative predictor
        if self.cumulative_predictor:
            num_features, max_followup = 4, 4 
            self.cum_prob_layer = Cumulative_Probability_Layer(num_features, max_followup=max_followup)


    def init_params(self):
        for p in self.parameters():
            if p.ndimension() >= 2:
                init.kaiming_normal(p)
            else:
                init.normal(p, mean=0.0, std=0.02)

    def printif(self, *args):
        if self.verbose:
            print(" ".join([str(a) for a in args]))

    def forward(self, x): #x_cond
        # first conv layer
        x = self.block0(x)
        self.printif("block0 out:", x.size())
        # block 1 --- subsample input and maxpool input for residual
        xmp = self.mp(x)
        xsub = x[:, :, ::2]
        xsub = (
            xsub[:, :, :-1] if x.shape[2] % 2 == 1 else xsub
        )  # block0 convolution adds a sample: to align, drop last subsample when x.shape[2] is odd.
        xsub = self.block1(xsub)
        x = xsub + xmp
        self.printif("block1 out:", x.size())

        # repeated blocks, apply
        for l, blk in enumerate(self.rep_blocks):
            if l % self.rep_mp == 0:  # maxpool every other layer
                xmp, xin = self.mp(x), x[:, :, :-1:2]
            else:
                xmp, xin = x, x
            x = blk(xin.contiguous())
            self.printif(l, "  repblock shape; xtmp shape ", x.shape, xmp.shape)
            x = x + xmp

        # fully connected out layer
        self.printif("before block_out", x.shape)
        x = self.block_out(x.contiguous())
        self.printif("after block_out", x.shape)
        self.printif("self.num_last_active", self.num_last_active)
        x = x.view(x.size()[0], self.num_last_active)
        self.printif("after view", x.shape)

        if self.cumulative_predictor and set(['scd3mo', 'scd3mo_6mo', 'scd6mo_1', 'scd1_2']).issubset(self.output_names):
            # 1) Indices of the interval-based inputs
            interval_names = ['scd3mo', 'scd3mo_6mo', 'scd6mo_1', 'scd1_2']
            interval_indices = [self.output_names.index(name) for name in interval_names]
            
            # 2) Extract these features from x
            x_intervals = torch.index_select(x, 1, torch.tensor(interval_indices, device=x.device))
            
            # 3) Pass them through the cumulative probability layer
            #    This should return shape (B, 4), each column is the cumulative risk at a specific horizon
            x_cumulative = self.cum_prob_layer(x_intervals)
            
            # 4) Indices of the final time horizons where we want the outputs
            #    e.g., scd3mo -> 3 month cumulative, scd6mo -> 6 month cumulative, etc.
            final_names = ['scd3mo', 'scd6mo', 'scd1', 'scd2']
            final_indices = [self.output_names.index(name) for name in final_names]
            
            # 5) Create a copy to hold updated values
            x_updated = x.clone()
            
            # 6) Insert the cumulative predictions into the correct final time-horizon slots
            for i, final_idx in enumerate(final_indices):
                indicator = torch.zeros_like(x_updated, device=x.device)
                indicator[:, final_idx] = 1
                x_updated = x_updated * (1 - indicator) + x_cumulative[:, i:i+1] * indicator
            
            x = x_updated

        return x


class BlockOne(nn.Module):
    """Resnet, first convolutional block"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0.5):
        """if subsample is true, then we subsample input and maxpool
        the residual term"""
        super(BlockOne, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=kernel_size // 2
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout1d(p=dropout),
        )
        self.conv = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )

    def forward(self, x):
        out = self.block(x)
        out = out[:, :, :-1]
        out = self.conv(out)
        out = out[:, :, :-1]
        return out


class RepBlock(nn.Module):
    """Resnet, Repeated Convolutional Block"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0.5):
        super(RepBlock, self).__init__()

        self.block1 = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Dropout1d(p=dropout),
            nn.Conv1d(
                in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=kernel_size // 2
            ),
        )

        self.block2 = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Dropout1d(p=dropout),
            nn.Conv1d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=kernel_size // 2
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout1d(p=dropout),
        )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = out[:, :, :-2]
        return out