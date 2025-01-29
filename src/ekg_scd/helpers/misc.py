from stringprep import in_table_c8
import typing
import os

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import numpy as np
import torch
from torch import nn

import torch.functional as F
from sklearn.metrics import average_precision_score, f1_score, precision_score, r2_score, recall_score, roc_auc_score

# used in some loss functions
ln2pi = np.log(2 * np.pi)

"""Implements performance metrics for our VAEs"""

from typing import Dict, Iterable, Callable

class VerboseExecution(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

        # Register a hook for each layer
        for name, layer in self.model.named_children():
            layer.__name__ = name
            layer.register_forward_hook(
                lambda layer, _, output: print(f"{layer.__name__}: {output.shape}")
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        _ = self.model(x)
        return self._features


def featurise(batch, model, layers):
    """Calculate features for given batch"""
    output = model(batch)

    featurise_fun = FeatureExtractor(model, layers=layers)
    features = featurise_fun(batch)

    return features


def beat_recon_bootstrap(x, recon_x, error_type="rmse", num_samples=100):
    if error_type == "rmse":
        err = np.sqrt(np.mean((x - recon_x) ** 2, axis=-1))
    elif error_type == "mae":
        err = np.mean(np.abs(x - recon_x), axis=-1)

    samps = []
    for _ in range(num_samples):
        idx = np.random.choice(len(err), size=len(err), replace=True)
        eboot = err[idx]
        samps.append(np.mean(eboot))

    return np.array(samps)


def bootstrap_auc(ytrue, ypred, fun=roc_auc_score, num_samples=100):
    # make nan safe
    nan_idx = np.isnan(ytrue)
    ytrue = ytrue[~nan_idx]
    ypred = ypred[~nan_idx]

    samps = []
    for _ in range(num_samples):
        idx = np.random.choice(len(ytrue), size=len(ytrue), replace=True)
        auc = fun(ytrue[idx], ypred[idx])
        # auc = np.max([auc, 1-auc])
        samps.append(auc)

    return np.array(samps)


def bootstrap_average_precision_score(ytrue, ypred, num_samples=100):
    return bootstrap_auc(ytrue, ypred, fun=average_precision_score, num_samples=num_samples)


def bootstrap_auc_comparison(ytrue, ypreda, ypredb, num_samples=100):
    samps_a, samps_b, diff = [], [], []
    for _ in range(num_samples):
        idx = np.random.choice(len(ytrue), size=len(ytrue), replace=True)
        auc_a = roc_auc_score(ytrue[idx], ypreda[idx])
        auc_b = roc_auc_score(ytrue[idx], ypredb[idx])
        samps_a.append(auc_a)
        samps_b.append(auc_b)
        diff.append(auc_a - auc_b)
    return samps_a, samps_b, diff


def bootstrap_prec_recall_f1(ytrue, ypred, num_samples=100):
    psamps, rsamps, fsamps = [], [], []
    for _ in range(num_samples):
        idx = np.random.choice(len(ytrue), size=len(ytrue), replace=True)
        psamps.append(precision_score(ytrue[idx], ypred[idx]))
        rsamps.append(recall_score(ytrue[idx], ypred[idx]))
        fsamps.append(f1_score(ytrue[idx], ypred[idx]))

    return np.array(psamps), np.array(rsamps), np.array(fsamps)


def bootstrap_corr(x, y, num_samples=100):
    samps = []
    for _ in range(num_samples):
        idx = np.random.choice(len(x), size=len(x), replace=True)
        samps.append(np.corrcoef(x[idx], y[idx])[0, 1])
    return np.array(samps)


def bootstrap_summary(y, fun=np.mean, num_samples=1000):
    samps = []
    for _ in range(num_samples):
        idx = np.random.choice(len(y), size=len(y), replace=True)
        samps.append(fun(y[idx]))
    return np.array(samps)


def bootstrap_r2(ytrue, ypred, num_samples=1000):
    samps = []
    for _ in range(num_samples):
        idx = np.random.choice(len(ytrue), size=len(ytrue), replace=True)
        samps.append(r2_score(ytrue[idx], ypred[idx]))
    return np.array(samps)


def plot_ekg(sampleEKG: np.array, sample_length: typing.Optional[int] = None,savepath: typing.Optional[str] = False, show:typing.Optional[bool]=False) -> None:
    """Function to plot a given 12-lead EKG record"""

    if sample_length is None:
        sample_length = sampleEKG.shape[1]

    time = [i for i in range(sample_length)]

    names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    fig, axs = plt.subplots(12, 1, figsize=(8, 8), facecolor="w", edgecolor="k")
    fig.subplots_adjust(hspace=1, wspace=0.001)
    fig.suptitle("Sample EKG")

    axs.ravel()

    for lead, num in zip(names, range(12)):
        axs[num].plot(time, sampleEKG[num, :], "-k", linewidth=1)
        axs[num].set(ylabel="%s /mV" % lead)
        # axs[num].set_yticks(np.arange(-2, 2, 1))

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels)
    plt.tight_layout()
    plt.rc('axes', titlesize=5)
    plt.xlabel("time/sample")
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath)
    if show:
        plt.show()
    plt.clf()

def plot_average_error_T(sampleEKG: np.array, sample_length: typing.Optional[int] = None) -> None:
    """Function to plot mean error conditional on T, and the 12-leads"""

    if sample_length is None:
        sample_length = sampleEKG.shape[1]

    time = [i for i in range(sample_length)]

    names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    fig, axs = plt.subplots(12, 1, figsize=(12, 12), facecolor="w", edgecolor="k")
    fig.subplots_adjust(hspace=1, wspace=0.001)
    fig.suptitle("Mean reconstruction error")
    fig.supylabel("mean(recon error | time/sample, lead)")

    axs.ravel()

    for lead, num in zip(names, range(12)):
        axs[num].plot(time, sampleEKG[num, :], "-k", linewidth=1)
        axs[num].set(ylabel="%s /mV" % lead)

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels)
    plt.xlabel("time/sample")
    plt.show()
    

def plot_error_bars(errors, exp_names: typing.Optional[list] = None) -> None:
    """Function to plot a given 12-lead EKG record"""

    n_channels = 12
    channel_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    fig, axs = plt.subplots(12, 1, figsize=(12, 12), facecolor="w", edgecolor="k")
    fig.subplots_adjust(hspace=1, wspace=0.001)
    fig.suptitle("Abs. Reconstruction Errors - varying dimensions of latent vector")

    axs.ravel()

    for lead, num in zip(channel_names, range(n_channels)):
        axs[num].plot(exp_names, errors.T[num])
        axs[num].set(ylabel="%s /mV" % lead)

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels)
    plt.xlabel("Experiment Value")


def plot_error_dist(errors, metric_name: typing.Optional[str] = None) -> None:
    """Function to plot distribution of a given test metric for across 12-leads"""

    n_channels = 12
    channel_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    fig, axs = plt.subplots(12, 1, figsize=(5, 15), facecolor="w", edgecolor="k")
    fig.subplots_adjust(hspace=1, wspace=0.001)
    fig.suptitle("Distribution of Test Metric %s" % metric_name)

    axs.ravel()

    for lead, num in zip(channel_names, range(n_channels)):
        axs[num].hist(errors.T.numpy()[num])
        axs[num].set(ylabel="Freq/ %s" % lead)

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels)
    plt.xlabel("difference in mV")


def plot_overlay(
    recons: np.array, actual: np.array, sample_length: typing.Optional[int] = None, i: typing.Optional[int] = None
):
    """Function to plot an estimates 12-lead EKG overlaid on true EKG series"""

    if sample_length is None:
        sample_length = recons.shape[1]

    time = [i for i in range(sample_length)]

    names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    fig, axs = plt.subplots(12, 1, figsize=(3, 12), facecolor="w", edgecolor="k")
    fig.subplots_adjust(hspace=1, wspace=0.001)
    fig.suptitle(f"EKG Sample with {i}_th Highest Recreation Error")

    axs.ravel()

    for lead, num in zip(names, range(12)):
        error = abs(actual[num, :] - recons[num, :])
        rmse = np.sqrt(np.sum(error**2))  # /max(v[num, :])
        axs[num].plot(time, actual[num, :], "-b", linewidth=1)
        axs[num].plot(time, recons[num, :], "--r", linewidth=1)

        axs[num].set(ylabel="%s /mV" % lead)

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels)
    plt.rc('axes', titlesize=5) 
    plt.tight_layout()
    plt.xlabel("time/sample")


def plot_diff(
    sampleEKG: np.array, sample_length: typing.Optional[int] = None, peaks: list = None, i: typing.Optional[int] = None
) -> None:
    """Function to plot the recnstruction error as a time series with annotations for highest magnitude "peaks" in the input."""

    import matplotlib.pyplot as plt

    if sample_length is None:
        sample_length = sampleEKG.shape[1]

    time = [i for i in range(sample_length)]

    names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    fig, axs = plt.subplots(12, 1, figsize=(12, 12), facecolor="w", edgecolor="k")
    fig.subplots_adjust(hspace=1, wspace=0.001)
    fig.suptitle("Time Series of Absolute Error in Record %i" % i)

    axs.ravel()

    for lead, num in zip(names, range(12)):
        axs[num].plot(time, sampleEKG[num, :], "-k", linewidth=1)
        axs[num].axvline(x=peaks[num], c="r")
        axs[num].set(ylabel="%s /mV" % lead)

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels)
    plt.xlabel("time/sample")


def pad_collate(data):
    """ECG beats are currently of non-uniform length. This function creates uniform-sized records in each batch"""
    # data = list(filter(lambda x: not torch.isnan(x[0]).any(), data))

    def merge(records):
        max_length = 512
        padded_recs = []

        for i, rec in enumerate(records):
            diff = max_length - rec.shape[1]
            if diff > 0:
                padded_rec = np.pad(rec, [(0, 0), (0, diff)], mode="mean")
            else:
                padded_rec = rec[:, :max_length]

            padded_rec = torch.Tensor(padded_rec)
            padded_recs.append(padded_rec)
        return padded_recs

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # seperate source and target sequences
    src_seqs, trg_seq = zip(*data)

    # batch signal data
    src_seqs = merge(src_seqs)
    src_seqs = torch.stack(src_seqs)

    trg_seq = [a[:3] for a in trg_seq]

    trg_seq = torch.stack(trg_seq)

    return src_seqs, trg_seq


def pairwise(list):
    i = iter(list)
    prev = i.__next__()
    for item in i:
        yield prev, item
        prev = item


def compare_models(model_1, model_2):
    """ Compare two models, assuming of the same structure, by searching and matching their state_dicts. """
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')