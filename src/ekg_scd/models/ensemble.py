""" Class for ensembling multiple DL architectures into one """

import numpy as np
import torch
import torch.nn as nn

from . import base


class EKGEnsembleModel(base.Model):
    def __init__(self, **kwargs):
        super(EKGEnsembleModel, self).__init__(**kwargs)
        self.model_list = kwargs.get("model_list")
        self.cat_size = sum([net.num_last_active for net in self.model_list])
        self.regress = np.array(kwargs.get("regress"))

        self.classify_lossfun = base.NanBCEWithLogitsLoss(reduction="mean")
        self.reg_lossfun = nn.MSELoss(size_average=True)

        self.n_outputs = kwargs.get("n_outputs")
        self.dropout = nn.Dropout(0.2)
        self.input_size = self.cat_size
        self.out = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_outputs),
        )

    def lossfun(self, data, target):
        logit = self.forward(data)
        if np.min(self.regress) == 1:
            pred_loss = self.reg_lossfun(logit, target)
        elif np.max(self.regress) == 0:
            pred_loss = self.classify_lossfun(logit, target)
        else:
            regression_loss = self.reg_lossfun(logit[:, self.regress], target[:, self.regress])
            classify_loss = self.classify_lossfun(logit[:, ~self.regress], target[:, ~self.regress])
            pred_loss = regression_loss * 10 + classify_loss
        return pred_loss, logit

    def forward(self, data):
        cat = self.model_list[0](data.clone())
        for m in range(1, len(self.model_list)):
            cat = torch.cat((cat, self.model_list[m](data.clone())), dim=1)
        # cat = self.dropout(cat) # Maybe see if this helps
        return self.out(cat)

    def fit(self, train, val, **kwargs):
        self.fit_res = base.fit_model(self, train, val, **kwargs)
        return self.fit_res


class EKGDeepWideEnsembleModel(EKGEnsembleModel):
    def __init__(self, **kwargs):
        super(EKGDeepWideEnsembleModel, self).__init__(**kwargs)
        self.dim_wide = kwargs.get("dim_wide")
        self.input_size = self.cat_size + self.dim_wide

    def lossfun(self, data, target):
        logit = self.forward(data)
        pred_loss = self.loss(logit, target)
        return torch.mean(pred_loss), logit

    def forward(self, data):
        """this module takes in a batch_sz x (C x T + dim_wide)"""
        wide = data[:, 0, -self.dim_wide :]

        # wide + EKG representation
        cat = self.model_list[0](data[:, :, : -self.dim_wide])
        for m in range(1, len(self.model_list)):
            cat = torch.cat((cat, self.model_list[m](data[:, :, : -self.dim_wide])), dim=1)
        # cat = self.dropout(cat)
        zout = torch.cat([cat, wide], 1)
        return self.out(zout)
