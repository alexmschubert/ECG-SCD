"""
Convert the pre-trained torch beat model to a JAX model
"""

import jax.numpy as jnp
from flax import linen as nn
import typing


class Block0(nn.Module):
    features: int
    kernel_size: int
    stride: int
    padding: int

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = nn.Conv(features=self.features, kernel_size=self.kernel_size, strides=self.stride, padding=self.padding)(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=0.9, epsilon=1e-05)(x)
        x = nn.relu(x)
        return x


class Block1(nn.Module):
    features: int
    kernel_size: int
    stride: int

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = nn.Conv(features=self.features, kernel_size=self.kernel_size, strides=self.stride, padding=self.kernel_size[0] // 2)(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=0.9, epsilon=1e-05)(x)
        x = nn.relu(x)
        x = nn.Dropout(0.5, deterministic=not train)(x)
        # lose dimension here (replicate bug from torch model)
        x = x[:, :-1, :]
        x = nn.Conv(features=self.features, kernel_size=self.kernel_size, strides=self.stride, padding=self.kernel_size[0] // 2)(x)
        # lose dimension here (replicate bug from torch model)
        x = x[:, :-1, :]
        return x


class RepBlock(nn.Module):
    features: int
    kernel_size: int
    stride: int

    @nn.compact
    def __call__(self, x, train: bool = True):
        # Block 1
        x1 = nn.BatchNorm(use_running_average=not train, momentum=0.9, epsilon=1e-05)(x)
        x1 = nn.relu(x1)
        x1 = nn.Dropout(0.5, deterministic=not train)(x1)
        x1 = nn.Conv(features=self.features, kernel_size=self.kernel_size, strides=self.stride, padding=self.kernel_size[0] // 2)(x1)

        # Block 2
        x2 = nn.BatchNorm(use_running_average=not train, momentum=0.9, epsilon=1e-05)(x1)
        x2 = nn.relu(x2)
        x2 = nn.Dropout(0.5, deterministic=not train)(x2)
        x2 = nn.Conv(features=self.features, kernel_size=self.kernel_size, strides=self.stride, padding=self.kernel_size[0] // 2)(x2)
        x2 = nn.BatchNorm(use_running_average=not train, momentum=0.9, epsilon=1e-05)(x2)
        x2 = nn.relu(x2)
        x2 = nn.Dropout(0.5, deterministic=not train)(x2)

        # lose final two layers (replicate bug from torch model)
        x2 = x2[:, :-2, :]
        return x2

class EKGResNetModel(nn.Module):
    num_rep_blocks: int = 32
    out_features: int = 3
    rep_mp: int = 4
    output_names: typing.Iterable[str] = ('scd3mo', 'scd6mo', 'scd1', 'scd2', 'dead3mo', 'dead6mo', 'dead1', 'dead2', 'ICD_VT1', 'ICD_VF1', 'Phi_VT1', 'TROPT1', 'Acute_MI1', 'age', 'female', 'qrsDuration', 'qtInterval', 'qrsFrontAxis', 'qrsHorizAxis', 'rrInterval', 'pDuration', 'atrialrate', 'meanqtc')

    @nn.compact
    def __call__(self, x, train: bool = True):
        # Block 0
        x = Block0(features=64, kernel_size=(16,), stride=(1,), padding=(8,))(x, train=train)

        # block 1 does some residual work too
        xmp = nn.max_pool(x, window_shape=(2,), strides=(2,), padding='VALID')
        xsub = x[:, ::2, :]
        # block0 convolution adds a sample: to align, drop last subsample when x.shape[2] is odd.
        xsub = (xsub[:, :-1, :] if x.shape[1] % 2 == 1 else xsub)  
        xsub = Block1(features=64, kernel_size=(16,), stride=(1,))(xsub, train=train)
        x = xsub + xmp

        for l in range(self.num_rep_blocks):
            if l % self.rep_mp == 0:  # maxpool every other layer
                xmp, xin = nn.max_pool(x, window_shape=(2,), strides=(2,), padding='VALID'), x[:, :-1:2, :]
            else:
                xmp, xin = x, x
            x = RepBlock(features=64, kernel_size=(16,), stride=(1,))(xin, train=train)
            x = x + xmp

        # Output block
        x = nn.BatchNorm(use_running_average=not train, momentum=0.1, epsilon=1e-05)(x)
        x = nn.relu(x)

        # Flattening and fully connected layer
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(self.out_features)(x)

        return x


def transform_key(pytorch_key):
    """Maps the pytorch key to the series of JAX keys---this is totally model specific"""
    # Split the key into parts
    parts = pytorch_key.split('.')
    if parts[-1] == "num_batches_tracked":
        return None
    elif parts[-1] in ["running_mean", "running_var", "num_batches_tracked"]:
        jax_key_0 = "batch_stats"
    else:
        jax_key_0 = "params"

    # create reference dictionary for layers
    layers_final_a = {'bias': 'bias', 'weight': 'kernel'}
    layers_final_b = {'bias': 'bias', 'weight': 'scale', "running_mean": "mean", "running_var": "var"}

    # iterate over parts of torch key
    # sometimes we need to grab the following key as well, so do this via index
    # block0:
    if parts[1] == "block0":
        conv_flag = parts[2] == "0"
        jax_key = [jax_key_0, "Block0_0", "Conv_0" if conv_flag else "BatchNorm_0", layers_final_a[parts[-1]] if conv_flag else layers_final_b[parts[-1]]]
        return jax_key

    # block 1:
    if parts[1] == "block1":
        if parts[2] == "block":
            conv_flag = parts[3] == "0"
            jax_key = [jax_key_0, "Block1_0", "Conv_0" if conv_flag else "BatchNorm_0", layers_final_a[parts[-1]] if conv_flag else layers_final_b[parts[-1]]]
            return jax_key
        elif parts[2] == "conv":
            return [jax_key_0, "Block1_0", "Conv_1", layers_final_a[parts[-1]]]
        else:
            raise ValueError
    
    # repblock:
    if parts[1] == "rep_blocks":
        jax_keys = [jax_key_0]
        # get block number
        block_n = parts[2]
        jax_keys.append("RepBlock_" + block_n)
        # get next part
        maps_next = {
            "block1.0": "BatchNorm_0",
            "block1.3": "Conv_0",
            "block2.0": "BatchNorm_1",
            "block2.3": "Conv_1",
            "block2.4": "BatchNorm_2"
        }
        jax_keys.append(maps_next[parts[3] + "." + parts[4]])
        jax_keys.append(layers_final_a[parts[-1]] if parts[4] == "3" else layers_final_b[parts[-1]])
        return jax_keys

    # final blocks--im tired just hard code
    if pytorch_key == "net.block_out.0.weight":
        return [jax_key_0, "BatchNorm_0", "scale"]
    elif pytorch_key == "net.block_out.0.bias":
        return [jax_key_0, "BatchNorm_0", "bias"]
    elif pytorch_key == "net.block_out.0.running_mean":
        return [jax_key_0, "BatchNorm_0", "mean"]
    elif pytorch_key == "net.block_out.0.running_var":
        return [jax_key_0, "BatchNorm_0", "var"]
    elif pytorch_key == "out.weight":
        return [jax_key_0, "Dense_0", "kernel"]
    elif pytorch_key == "out.bias":
        return [jax_key_0, "Dense_0", "bias"]

    else:
        raise ValueError(f"Unrecognized PyTorch key: {pytorch_key}")


def pytorch_to_jax(pytorch_state_dict):
    """Extracts coefficients from pytorch dict and creates a JAX parameter dictionary
    
    Uses jax_reference_dict to check that keys in the jax dictionary exist.
    """
    jax_state_dict = {}
    for k, v in pytorch_state_dict.items():
        # Transform the key to match JAX naming and structure
        # This step is highly specific to the model architecture and naming convention
        jax_key = transform_key(k)  # Implement this function based on your model's parameter naming

        # none corresponds to parameters not yet tracked by JAX, skip
        if jax_key is None:
            continue

        # Convert PyTorch tensor to JAX/numpy array
        jax_value = jnp.array(v.cpu().numpy())

        # reshape coefficients
        if any("Conv" in key for key in jax_key) and jax_key[-1] == "kernel":
            jax_value = jax_value.transpose((2, 1, 0))
        elif any("Dense" in key for key in jax_key) and jax_key[-1] == "kernel":
            jax_value = jax_value.transpose((1, 0))

        dict_last = jax_state_dict
        for key in jax_key[:-1]:
            if key not in dict_last:
                dict_last[key] = {}
            dict_last = dict_last[key]

        dict_last[jax_key[-1]] = jax_value

    return jax_state_dict

