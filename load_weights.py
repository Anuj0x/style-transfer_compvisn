"""
Utility to load pretrained VGG19 weights from PyTorch and convert to JAX format.
This enables proper initialization of the JAX VGG19 model with pretrained weights.
"""

import torch
import torchvision.models as models
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Any
import numpy as np


def load_pytorch_vgg19_weights() -> Dict[str, Any]:
    """
    Load pretrained VGG19 weights from torchvision and convert to JAX format.

    Returns:
        Dictionary of JAX-compatible weight arrays
    """
    # Load pretrained VGG19 from PyTorch
    pytorch_model = models.vgg19(pretrained=True)
    state_dict = pytorch_model.state_dict()

    jax_weights = {}

    # VGG19 layer mapping for our custom model
    layer_mapping = {
        # Block 1
        'Conv_0': 'features.0',  # conv1_1
        'Conv_1': 'features.2',  # conv1_2

        # Block 2
        'Conv_2': 'features.5',  # conv2_1
        'Conv_3': 'features.7',  # conv2_2

        # Block 3
        'Conv_4': 'features.10',  # conv3_1
        'Conv_5': 'features.12',  # conv3_2
        'Conv_6': 'features.14',  # conv3_3

        # Block 4
        'Conv_7': 'features.17',  # conv4_1
        'Conv_8': 'features.19',  # conv4_2
        'Conv_9': 'features.21',  # conv4_3
        'Conv_10': 'features.23',  # conv4_4

        # Block 5
        'Conv_11': 'features.26',  # conv5_1
        'Conv_12': 'features.28',  # conv5_2
        'Conv_13': 'features.30',  # conv5_3
    }

    for jax_name, pytorch_name in layer_mapping.items():
        # Load weights and bias
        weight_key = f'{pytorch_name}.weight'
        bias_key = f'{pytorch_name}.bias'

        if weight_key in state_dict and bias_key in state_dict:
            # Convert PyTorch tensors to JAX arrays
            # PyTorch: (out_channels, in_channels, kernel_h, kernel_w)
            # JAX/Flax: (kernel_h, kernel_w, in_channels, out_channels)
            pytorch_weight = state_dict[weight_key].numpy()
            jax_weight = np.transpose(pytorch_weight, (2, 3, 1, 0))

            jax_weights[f'{jax_name}/kernel'] = jnp.array(jax_weight)
            jax_weights[f'{jax_name}/bias'] = jnp.array(state_dict[bias_key].numpy())

    return jax_weights


def create_pretrained_variables(model: nn.Module, jax_weights: Dict[str, jnp.ndarray]) -> Dict:
    """
    Create Flax variables with pretrained weights.

    Args:
        model: Flax model instance
        jax_weights: Dictionary of pretrained weights

    Returns:
        Flax variables dictionary
    """
    # Initialize with dummy input to get parameter structure
    dummy_input = jnp.ones((1, 224, 224, 3))
    variables = model.init(jax.random.PRNGKey(0), dummy_input)

    # Replace with pretrained weights
    pretrained_params = {}
    for key, value in variables['params'].items():
        if isinstance(value, dict):
            pretrained_params[key] = {}
            for subkey, subvalue in value.items():
                param_key = f'{key}/{subkey}'
                if param_key in jax_weights:
                    pretrained_params[key][subkey] = jax_weights[param_key]
                    print(f"Loaded pretrained weights for {param_key}")
                else:
                    pretrained_params[key][subkey] = subvalue
                    print(f"Using random initialization for {param_key}")
        else:
            param_key = key
            if param_key in jax_weights:
                pretrained_params[key] = jax_weights[param_key]
                print(f"Loaded pretrained weights for {param_key}")
            else:
                pretrained_params[key] = value
                print(f"Using random initialization for {param_key}")

    return {'params': pretrained_params}


def load_pretrained_vgg19_with_weights(model: nn.Module) -> tuple:
    """
    Load VGG19 model with pretrained weights.

    Args:
        model: VGG19Features model instance

    Returns:
        Tuple of (model, variables) with pretrained weights
    """
    print("Loading pretrained VGG19 weights...")
    jax_weights = load_pytorch_vgg19_weights()
    variables = create_pretrained_variables(model, jax_weights)
    print("Pretrained weights loaded successfully!")
    return model, variables


if __name__ == "__main__":
    # Test weight loading
    from nst_jax import VGG19Features

    model = VGG19Features()
    model, variables = load_pretrained_vgg19_with_weights(model)
    print("Model loaded with pretrained weights!")
