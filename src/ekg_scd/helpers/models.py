""" Functions related to the establishment and management of DL models. """

import torch
from ekg_scd.models import resnet


def establish_params(param_X, param_y):
    """Establish parameters for training"""

    dim_wide = 0
    n_channels = param_X.shape[0]
    n_samples = param_X.shape[1] - dim_wide
    n_outputs = param_y.shape[0]

    return n_channels, n_samples, n_outputs, dim_wide


def establish_model(
    model, regress, n_channels, n_samples, n_outputs, dim_wide, ensemble_models=None, ensemble_paths=None, cumulative_predictor=False, 
    output_names= None, dropout=0.5, covariate_conditioning = None, auxiliary_layers = None, num_rep_blocks = 32, kernel_size = 16, rep_mp = 4,
    attention = False, conv_channels=64
    ,

):
    """Establish model class with provided parameters."""

    # Establish ResNet params
    deepwide = "DeepWide" if dim_wide > 0 else ""
    aux = "Aux" if auxiliary_layers else ""
    name = f"EKG{deepwide}{model}{aux}Model"

    # Load model ------------
    if model == "ResNet":
        method_to_call = getattr(resnet, name)
        m = method_to_call(
            n_channels=n_channels,
            n_samples=n_samples,
            n_outputs=n_outputs,
            dim_wide=dim_wide,
            num_rep_blocks=num_rep_blocks,  # Was 15
            kernel_size=kernel_size,
            verbose=False,
            regress=regress,
            cumulative_predictor = cumulative_predictor,
            output_names = output_names,
            dropout = dropout,
            covariate_conditioning = covariate_conditioning,
            auxiliary_layers = auxiliary_layers,
            rep_mp = rep_mp,
            attention = attention,
            conv_channels = conv_channels
        )  # To set loss

    return m


def load_model_with_path(model, path):
    """
    Given instantiated model class and fitted model path, load weights.
    """

    state = torch.load(f"{path}/model.best.pth.tar")["best_state"]
    state = {key.replace("net.", ""): val for key, val in state.items()}
    state.pop("out.weight", None)
    state.pop("out.bias", None)
    model.load_state_dict(state)
    model.fix_params()

    return model
