"""
Modern Neural Style Transfer Implementation using JAX
=====================================================

A high-performance, elegant implementation of Neural Style Transfer using JAX/Flax.
Features JIT compilation, vectorization, and modern functional programming paradigms.

Author: Cline (AI Assistant)
Based on Gatys et al. (2016) with modern optimizations
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import numpy as np
import cv2
import os
from typing import Dict, Tuple, List, Optional
from functools import partial
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
IMAGENET_MEAN = jnp.array([0.485, 0.456, 0.406])
IMAGENET_STD = jnp.array([0.229, 0.224, 0.225])


class VGG19Features(nn.Module):
    """
    VGG19 feature extractor optimized for style transfer.
    Uses only the layers needed for content and style representation.
    """

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        features = {}

        # Block 1
        x = nn.Conv(64, (3, 3), padding=1)(x)
        x = nn.relu(x)
        features['relu1_1'] = x

        x = nn.Conv(64, (3, 3), padding=1)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2), strides=(2, 2))

        # Block 2
        x = nn.Conv(128, (3, 3), padding=1)(x)
        x = nn.relu(x)
        features['relu2_1'] = x

        x = nn.Conv(128, (3, 3), padding=1)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2), strides=(2, 2))

        # Block 3
        x = nn.Conv(256, (3, 3), padding=1)(x)
        x = nn.relu(x)
        x = nn.Conv(256, (3, 3), padding=1)(x)
        x = nn.relu(x)
        x = nn.Conv(256, (3, 3), padding=1)(x)
        x = nn.relu(x)
        features['relu3_1'] = x

        x = nn.max_pool(x, (2, 2), strides=(2, 2))

        # Block 4
        x = nn.Conv(512, (3, 3), padding=1)(x)
        x = nn.relu(x)
        x = nn.Conv(512, (3, 3), padding=1)(x)
        x = nn.relu(x)
        x = nn.Conv(512, (3, 3), padding=1)(x)
        x = nn.relu(x)
        features['relu4_1'] = x

        x = nn.Conv(512, (3, 3), padding=1)(x)
        x = nn.relu(x)
        features['conv4_2'] = x  # Content representation

        x = nn.max_pool(x, (2, 2), strides=(2, 2))

        # Block 5
        x = nn.Conv(512, (3, 3), padding=1)(x)
        x = nn.relu(x)
        x = nn.Conv(512, (3, 3), padding=1)(x)
        x = nn.relu(x)
        x = nn.Conv(512, (3, 3), padding=1)(x)
        x = nn.relu(x)
        features['relu5_1'] = x

        return features


def load_pretrained_vgg19() -> VGG19Features:
    """
    Load VGG19 with pretrained weights from torchvision.
    """
    from load_weights import load_pretrained_vgg19_with_weights

    model = VGG19Features()
    model, variables = load_pretrained_vgg19_with_weights(model)
    return model, variables


def preprocess_image(image: jnp.ndarray, target_size: Optional[int] = None) -> jnp.ndarray:
    """Preprocess image for VGG19 input."""
    # Resize if needed
    if target_size is not None:
        # Simple bilinear resize (in practice, use proper image resizing)
        h, w = image.shape[:2]
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = jax.image.resize(image, (new_h, new_w, 3), method='bilinear')

    # Convert to float and normalize to [0, 1]
    image = image.astype(jnp.float32) / 255.0

    # Normalize with ImageNet statistics
    image = (image - IMAGENET_MEAN) / IMAGENET_STD

    return image

# JIT-compiled version for efficiency
preprocess_image_jit = jax.jit(preprocess_image)


def gram_matrix(feature_map: jnp.ndarray) -> jnp.ndarray:
    """Compute Gram matrix for style representation."""
    b, h, w, c = feature_map.shape
    features = feature_map.reshape(b, h * w, c)
    gram = jnp.einsum('bik,bjk->bij', features, features)
    return gram / (h * w * c)


@jax.jit
def content_loss(content_features: jnp.ndarray, target_features: jnp.ndarray) -> jnp.ndarray:
    """Compute content loss using MSE."""
    return jnp.mean((content_features - target_features) ** 2)


@jax.jit
def style_loss(style_grams: List[jnp.ndarray], target_grams: List[jnp.ndarray]) -> jnp.ndarray:
    """Compute style loss across multiple layers."""
    total_loss = 0.0
    for style_gram, target_gram in zip(style_grams, target_grams):
        total_loss += jnp.mean((style_gram - target_gram) ** 2)
    return total_loss / len(style_grams)


@jax.jit
def total_variation_loss(image: jnp.ndarray) -> jnp.ndarray:
    """Compute total variation loss for spatial smoothness."""
    # Compute horizontal and vertical differences
    diff_h = image[:, :-1, :, :] - image[:, 1:, :, :]
    diff_v = image[:, :, :-1, :] - image[:, :, 1:, :]

    return jnp.sum(diff_h ** 2) + jnp.sum(diff_v ** 2)


@jax.jit
def total_loss_fn(generated_features: Dict[str, jnp.ndarray],
                  content_target: jnp.ndarray,
                  style_targets: List[jnp.ndarray],
                  generated_image: jnp.ndarray,
                  weights: Dict[str, float]) -> jnp.ndarray:
    """Compute total loss for optimization."""

    # Content loss
    content_loss_val = content_loss(generated_features['conv4_2'], content_target)

    # Style loss
    style_grams = [gram_matrix(generated_features[layer])
                   for layer in ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']]
    style_loss_val = style_loss(style_grams, style_targets)

    # Total variation loss
    tv_loss_val = total_variation_loss(generated_image)

    # Weighted sum
    total = (weights['content'] * content_loss_val +
             weights['style'] * style_loss_val +
             weights['tv'] * tv_loss_val)

    return total, {
        'content': content_loss_val,
        'style': style_loss_val,
        'tv': tv_loss_val
    }


def create_optimizer(lr: float = 1.0):
    """Create L-BFGS optimizer (simplified version)."""
    # Note: JAX doesn't have L-BFGS built-in, using Adam as approximation
    # For true L-BFGS, you'd need a custom implementation
    return optax.adam(lr)


@partial(jax.jit, static_argnums=(0, 1, 5))
def optimization_step(model: VGG19Features,
                     variables: Dict,
                     content_target: jnp.ndarray,
                     style_targets: List[jnp.ndarray],
                     opt_state: train_state.TrainState,
                     weights: Dict[str, float]):
    """Single optimization step."""

    def loss_fn(generated_image):
        features = model.apply(variables, generated_image)
        loss, loss_components = total_loss_fn(
            features, content_target, style_targets, generated_image, weights
        )
        return loss, loss_components

    (loss, loss_components), grads = jax.value_and_grad(loss_fn, has_aux=True)(opt_state.params)

    opt_state = opt_state.apply_gradients(grads=grads)

    return opt_state, loss, loss_components


def neural_style_transfer(content_path: str,
                         style_path: str,
                         output_path: str,
                         config: Dict) -> str:
    """
    Perform neural style transfer with modern JAX implementation.

    Args:
        content_path: Path to content image
        style_path: Path to style image
        output_path: Output directory path
        config: Configuration dictionary

    Returns:
        Path to generated image
    """

    logger.info("Starting Neural Style Transfer...")

    # Load and preprocess images
    content_img = cv2.imread(content_path)
    style_img = cv2.imread(style_path)

    if content_img is None or style_img is None:
        raise ValueError("Could not load input images")

    # Convert BGR to RGB
    content_img = cv2.cvtColor(content_img, cv2.COLOR_BGR2RGB)
    style_img = cv2.cvtColor(style_img, cv2.COLOR_BGR2RGB)

    # Preprocess images
    content_processed = preprocess_image(content_img, config.get('height'))
    style_processed = preprocess_image(style_img, config.get('height'))

    # Add batch dimension
    content_processed = content_processed[None, ...]  # (1, H, W, 3)
    style_processed = style_processed[None, ...]

    # Initialize model
    model, variables = load_pretrained_vgg19()

    # Extract target features
    logger.info("Extracting target features...")
    content_features = model.apply(variables, content_processed)
    style_features = model.apply(variables, style_processed)

    content_target = content_features['conv4_2']
    style_targets = [gram_matrix(style_features[layer])
                    for layer in ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']]

    # Initialize generated image (start with content image)
    generated_image = content_processed.copy()

    # Setup optimizer
    optimizer = create_optimizer(config.get('learning_rate', 1.0))
    opt_state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=generated_image,
        tx=optimizer
    )

    # Optimization weights
    weights = {
        'content': config.get('content_weight', 1e6),
        'style': config.get('style_weight', 1e4),
        'tv': config.get('tv_weight', 1e-6)
    }

    # Optimization loop
    num_iterations = config.get('num_iterations', 1000)
    logger.info(f"Starting optimization for {num_iterations} iterations...")

    for i in range(num_iterations):
        opt_state, loss, loss_components = optimization_step(
            model, variables, content_target, style_targets, opt_state, weights
        )

        if i % 100 == 0:
            logger.info(f"Iteration {i:4d}: Total Loss = {loss:.4f}, "
                       f"Content = {loss_components['content']:.4f}, "
                       f"Style = {loss_components['style']:.4f}, "
                       f"TV = {loss_components['tv']:.4f}")

    # Save result
    result_image = opt_state.params[0]  # Remove batch dimension

    # Denormalize
    result_image = result_image * IMAGENET_STD + IMAGENET_MEAN
    result_image = jnp.clip(result_image, 0, 1)

    # Convert to uint8
    result_image = (result_image * 255).astype(jnp.uint8)

    # Convert to numpy for OpenCV
    result_np = np.array(result_image)

    # Save result
    output_filename = f"{os.path.basename(content_path).split('.')[0]}_{os.path.basename(style_path).split('.')[0]}_jax.jpg"
    output_full_path = os.path.join(output_path, output_filename)

    cv2.imwrite(output_full_path, cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR))
    logger.info(f"Result saved to: {output_full_path}")

    return output_full_path


def main():
    """Main function for command-line usage."""

    # Configuration
    config = {
        'height': 400,
        'content_weight': 1e6,
        'style_weight': 1e4,
        'tv_weight': 1e-6,
        'num_iterations': 500,  # Reduced for faster demo
        'learning_rate': 0.01
    }

    # Paths
    base_path = 'data'
    content_dir = os.path.join(base_path, 'content-images')
    style_dir = os.path.join(base_path, 'style-images')
    output_dir = os.path.join(base_path, 'output-images')

    os.makedirs(output_dir, exist_ok=True)

    # Example usage
    content_img = 'c1.jpg'
    style_img = 's1.jpg'

    content_path = os.path.join(content_dir, content_img)
    style_path = os.path.join(style_dir, style_img)

    if not os.path.exists(content_path) or not os.path.exists(style_path):
        logger.error(f"Content or style image not found: {content_path}, {style_path}")
        return

    # Run style transfer
    result_path = neural_style_transfer(content_path, style_path, output_dir, config)
    logger.info(f"Style transfer completed! Result: {result_path}")


if __name__ == "__main__":
    main()
