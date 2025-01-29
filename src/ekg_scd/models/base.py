""" base model structure from which all other inherit

provides all utilities for model fitting apart from model architecture
"""

import copy
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, r2_score, roc_auc_score
from torch import nn, optim
from torch.autograd import Variable
from tqdm import tqdm
#import wandb

def roc_auc_score_nan(y, p):
    nan_idx = pd.isnull(y) | pd.isnull(p)
    try:
        score = roc_auc_score(y[~nan_idx], p[~nan_idx])
    except:
        score = float("NaN")
    return score


def f1_score_nan(y, p):
    nan_idx = np.isnan(y)
    return f1_score(y[~nan_idx], p[~nan_idx])


def r2_score_nan(y, p):
    nan_idx = pd.isnull(y) | pd.isnull(p)
    try:
        score = r2_score(y[~nan_idx], p[~nan_idx])
    except:
        score = float("NaN")
    return score


class Model(nn.Module):
    """base model class w/ some helper functions for training/manipulating
    parameters, and saving
    """

    def __init__(self, **kwargs):
        super(Model, self).__init__()
        self.kwargs = kwargs
        self.fit_res = None

    def save(self, filename):
        torch.save(
            {
                "state_dict": self.state_dict(),
                "kwargs": self.kwargs,
                "fit_res": self.fit_res,
                "model_class": type(self),
            },
            f=filename,
        )

    def fit(self, data):
        raise NotImplementedError

    def lossfun(self, data, target):
        raise NotImplementedError

    def init_params(self):
        for p in self.parameters():
            if p.requires_grad == True:
                p.data.uniform_(-0.05, 0.05)

    def fix_params(self):
        for p in self.parameters():
            p.requires_grad = False

    def free_params(self):
        for p in self.parameters():
            p.requires_grad = True

    def num_params(self):
        return np.sum([p.numel() for p in self.parameters()])


def load_model(fname):
    model_dict = torch.load(fname)
    mod = model_dict["model_class"](**model_dict["kwargs"])
    mod.load_state_dict(model_dict["state_dict"])
    mod.fit_res = model_dict["fit_res"]
    return mod


class MaskedBCELoss(nn.Module):
    """BCELoss that accounts for NaNs (given by mask)"""

    def __init__(self, reduction="mean"):
        super(MaskedBCELoss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction=reduction)

    def forward(self, output, target, mask):
        """masked binary cross entropy loss
        Args:
          - output: batch_size x D float tensor with values in [0, 1]
          - target: batch_size x D float tensor with values in {0, 1}
          - mask  : batch_size x D byte tensor with 1 = not nan (include in loss)
        """
        tvec = target.view(-1)
        ovec = output.view(-1)
        mvec = mask.view(-1)

        # grab valid --- return bce loss
        tvalid = tvec.masked_select(mvec)
        ovalid = ovec.masked_select(mvec)
        return self.bce_loss(ovalid, tvalid)


def isnan(x):
    return x != x


class NanBCEWithLogitsLoss(nn.Module):
    """BCELoss that accounts for NaNs (given by mask)"""

    def __init__(self, reduction="mean"):
        super(NanBCEWithLogitsLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, output, target):
        """masked binary cross entropy loss
        Args:
          - output: batch_size x D float tensor with values in [0, 1]
          - target: batch_size x D float tensor with values in {0, 1}
          - mask  : batch_size x D byte tensor with 1 = not nan (include in loss)
        """
        tvec = target.view(-1)
        ovec = output.view(-1)
        mvec = Variable(~isnan(tvec).data)

        # grab valid --- return bce loss
        tvalid = tvec.masked_select(mvec)
        ovalid = ovec.masked_select(mvec)
        return self.bce_loss(ovalid, tvalid)


class NanMSELoss(nn.Module):
    """BCELoss that accounts for NaNs (given by mask)"""

    def __init__(self, reduction="mean"):
        super(NanMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="mean")

    def forward(self, output, target):
        """masked binary cross entropy loss
        Args:
          - output: batch_size x D float tensor with values in [0, 1]
          - target: batch_size x D float tensor with values in {0, 1}
          - mask  : batch_size x D byte tensor with 1 = not nan (include in loss)
        """
        tvec = target.view(-1)
        ovec = output.view(-1)
        mvec = Variable(~isnan(tvec).data)

        # grab valid --- return bce loss
        tvalid = tvec.masked_select(mvec)
        ovalid = ovec.masked_select(mvec)
        return self.mse_loss(ovalid, tvalid)


##############################
# standard fitting procedure #
##############################


def fit_model(model, train_loader, val_loader, **kwargs):

    # [OPTIONAL] log model performance using wandb
    # wandb.init(project="ECG-Halland",
    #         #name="your_run_name",
    #         entity="your_entity_name",
    #         config=kwargs)

    do_cuda = kwargs.get("do_cuda", torch.cuda.is_available())
    min_epochs = kwargs.get("min_epochs", 40)
    max_epochs = kwargs.get("max_epochs", 100)
    patience = kwargs.get("patience", 10)
    weight_decay = kwargs.get("weight_decay", 1e-5)
    learning_rate = kwargs.get("learning_rate", 1e-2)
    lr_reduce_interval = kwargs.get("lr_reduce_interval", 10)
    lr_sched_gamma = kwargs.get("lr_sched_gamma", 0.5)
    opt_type = kwargs.get("optimizer", "adam")
    log_interval = kwargs.get("log_interval", False)
    warm_start_path = kwargs.get("warm_start_path", None)
    warm_start_from_zero = kwargs.get("warm_start_from_zero", False)
    save_path = kwargs.get("save_path", None)
    covariate_conditioning = kwargs.get("covariate_conditioning",)

    print("-------------------")
    print("fitting model: ", kwargs)

    # set up optimizer
    plist = list(filter(lambda p: p.requires_grad, model.parameters()))
    if opt_type == "adam":
        optimizer = optim.Adam(plist, lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(plist, lr=learning_rate, weight_decay=weight_decay)

    if do_cuda:
        torch.cuda.set_device("cuda:0")
        model.cuda()

    model_dict = {}
    model_dict["train_loss"] = []
    model_dict["val_loss"] = []
    model_dict["best_loss"] = np.inf
    model_dict["best_epoch"] = 0
    start_epoch = 0

    if warm_start_path:
        checkpoint_dict = torch.load(warm_start_path)

        if warm_start_from_zero:
            start_dict = model.state_dict()
            init_dict = {k: v for k, v in checkpoint_dict["best_state"].items() if k not in ["out.weight", "out.bias", "out.2.weight", "out.2.bias", "out.0.weight", "out.0.bias"]}
            start_dict.update(init_dict)
            model.load_state_dict(start_dict)

        else:
            # Start up from previous state and epoch
            model.load_state_dict(checkpoint_dict["stop_state"])
            optimizer.load_state_dict(checkpoint_dict["stop_optimizer"])
            start_epoch = checkpoint_dict["stop_epoch"] + 1

            # Establish performance stats
            model_dict = {
                key: checkpoint_dict[key]
                for key in ["best_epoch", "best_loss", "best_stat", "best_state", "train_loss", "val_loss"]
            }

    for epoch in range(start_epoch, max_epochs):

        tloss, tstat = run_epoch(
            epoch, model, train_loader, optimizer, do_cuda, only_compute_loss=False, log_interval=log_interval, covariate_conditioning=covariate_conditioning
        )
        vloss, vstat = run_epoch(
            epoch, model, val_loader, optimizer, do_cuda, only_compute_loss=True, log_interval=log_interval, covariate_conditioning=covariate_conditioning
        )

        # [optional] Log metrics to wandb
        # wandb.log({"train_loss": tloss, "val_loss": vloss, 'val_auc': float(vstat[0]), "epoch": epoch})

        print(f"Loss ({epoch}): {round(vloss,6)}, Stat: {[round(float(v),6) for v in vstat]}")

        model_dict["train_loss"].append(tloss)
        model_dict["val_loss"].append(vloss)

        if vloss < model_dict["best_loss"]:
            print("  (updating best loss)")
            model_dict["best_loss"] = vloss
            model_dict["best_stat"] = vstat

            # Save new best parameters
            model_dict["best_epoch"] = epoch
            model_dict["best_state"] = copy.deepcopy(model.state_dict())

        warm_start_dict = {
            "stop_epoch": epoch,
            "stop_state": copy.deepcopy(model.state_dict()),
            "stop_optimizer": copy.deepcopy(optimizer.state_dict()),
        }

        if (epoch >= model_dict["best_epoch"] + patience and epoch >= min_epochs) or epoch == max_epochs:
            print(f"Failed to improve loss for {patience} epochs. Stopping at epoch {epoch}")
            break

        # Save every epoch so that we can warm_start from the last epoch
        torch.save({**model_dict, **warm_start_dict}, f=save_path + "/current.best.pth.tar")

        if epoch % lr_reduce_interval == 0 and epoch != 0:
            print("... reducing learning rate!")
            for param_group in optimizer.param_groups:
                param_group["lr"] *= lr_sched_gamma

    torch.save(model_dict, f=save_path + "/model.best.pth.tar")
    os.remove(save_path + "/current.best.pth.tar")

    print(f"Training complete! Outputs and hyperparameters saved to {save_path}")


def run_epoch(epoch, model, data_loader, optimizer, do_cuda, 
              only_compute_loss=False, log_interval=20, covariate_conditioning= False):
    if only_compute_loss:
        model.eval()
    else:
        model.train()

    # iterate over batches
    total_loss = 0
    trues, preds = [], []
    print(f"Enumerating batches, (epoch {epoch})")
    if covariate_conditioning:
        for batch_idx, (data, cov, target) in enumerate(tqdm(data_loader)):
            data, cov, target = Variable(data), Variable(cov),Variable(target)
            if do_cuda:
                data, cov, target = data.cuda(), cov.cuda(), target.cuda()
                data, cov, target = data.contiguous(), cov.cuda(), target.contiguous()

            # set up optimizer
            if not only_compute_loss:
                optimizer.zero_grad()

            # push data through model (make sure the recon batch batches data)
            loss, logitpreds = model.lossfun(data, target, cov)

            # backprop
            if not only_compute_loss:
                with torch.autograd.set_detect_anomaly(True):
                    loss.backward()
                    optimizer.step()

            # track pred probs
            logitpreds[:, ~model.regress] = torch.sigmoid(logitpreds[:, ~model.regress])

            trues.append(target.data.cpu().numpy())
            preds.append(logitpreds.data.cpu().numpy())
            total_loss += loss.item()
            if (log_interval != False) and (batch_idx % log_interval == 0):
                print(
                    "{pre} Epoch: {ep} [{cb}/{tb} ({frac:.0f}%)]\tLoss: {loss:.6f}".format(
                        pre="  Val" if only_compute_loss else "  Train",
                        ep=epoch,
                        cb=batch_idx * data_loader.batch_size,
                        tb=len(data_loader.dataset),
                        frac=100.0 * batch_idx / len(data_loader),
                        loss=total_loss / (batch_idx + 1),
                    )
                )

            # To prevent memory errors
            del loss, data, target, logitpreds
            torch.cuda.empty_cache()
    else:
        for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
            data, target = Variable(data), Variable(target)
            if do_cuda:
                data, target = data.cuda(), target.cuda()
                data, target = data.contiguous(), target.contiguous()

            # set up optimizer
            if not only_compute_loss:
                optimizer.zero_grad()

            # push data through model (make sure the recon batch batches data)
            loss, logitpreds = model.lossfun(data, target)

            # backprop
            if not only_compute_loss:
                loss.backward()
                optimizer.step()

            # track pred probs
            logitpreds[:, ~model.regress] = torch.sigmoid(logitpreds[:, ~model.regress])

            trues.append(target.data.cpu().numpy())
            preds.append(logitpreds.data.cpu().numpy())
            total_loss += loss.item()
            if (log_interval != False) and (batch_idx % log_interval == 0):
                print(
                    "{pre} Epoch: {ep} [{cb}/{tb} ({frac:.0f}%)]\tLoss: {loss:.6f}".format(
                        pre="  Val" if only_compute_loss else "  Train",
                        ep=epoch,
                        cb=batch_idx * data_loader.batch_size,
                        tb=len(data_loader.dataset),
                        frac=100.0 * batch_idx / len(data_loader),
                        loss=total_loss / (batch_idx + 1),
                    )
                )

            # To prevent memory errors
            del loss, data, target, logitpreds
            torch.cuda.empty_cache()

    total_loss /= len(data_loader)
    trues, preds = np.row_stack(trues), np.row_stack(preds)

    # Calculate relevant statistic
    stats = [
        r2_score_nan(true, pred) if r == True else roc_auc_score_nan(true, pred)
        for true, pred, r in zip(trues.T, preds.T, model.regress)
    ]

    return total_loss, stats