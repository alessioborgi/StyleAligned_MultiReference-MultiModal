"""
Encode_Image.py

This file contains the implementation of the Image Encoding function.
Authors:
- Alessio Borgi (alessioborgi3@gmail.com)
- Francesco Danese (danese.1926188@studenti.uniroma1.it)

Created on: July 6, 2024
"""


from __future__ import annotations
import torch
import numpy as np
from src.Handler import Handler
from diffusers import StableDiffusionXLPipeline
from src.StyleAlignedArgs import StyleAlignedArgs


def image_encoding(model: StableDiffusionXLPipeline, image: np.ndarray) -> T:

    # 1) Set VAE to Float32: Ensure the VAE operates in float32 precision for encoding.
    model.vae.to(dtype=torch.float32)

    # 2) Convert Image to PyTorch Tensor: Convert the input image from a numpy array to a PyTorch tensor and normalize pixel values to [0, 1].
    scaled_image = torch.from_numpy(image).float() / 255.

    # 3) Normalize and Prepare Image: Scale pixel values to the range [-1, 1], rearrange dimensions, and add batch dimension.
    permuted_image = (scaled_image * 2 - 1).permute(2, 0, 1).unsqueeze(0)

    # 4) Encode Image Using VAE: Use the VAE to encode the image into the latent space.
    latent_img = model.vae.encode(permuted_image.to(model.vae.device))['latent_dist'].mean * model.vae.config.scaling_factor

    # 5) Reset VAE to Float16: Optionally reset the VAE to float16 precision.
    model.vae.to(dtype=torch.float16)

    # 6) Return Latent Representation: Return the encoded latent representation of the image.
    return latent_img


### LINEAR WEIGHTED AVERAGE #################################
def images_encoding(model, images: list[np.ndarray], blending_weights: list[float]):
    """
    Encode a list of images using the VAE model and blend their latent representations
    according to the given blending_weights.

    Args:
    - model: The StableDiffusionXLPipeline model.
    - images: A list of numpy arrays, each representing an image.
    - blending_weights: A list of floats representing the blending weights for each image.
              The blending_weights should sum to 1.

    Returns:
    - blended_latent_img: The blended latent representation.
    """

    # Ensure the blending_weights sum to 1.
    assert len(images) == len(blending_weights), "The number of images and blending_weights must match."
    assert np.isclose(sum(blending_weights), 1.0), "blending_weights must sum to 1."

    # Set VAE to Float32 for encoding.
    model.vae.to(dtype=torch.float32)

    # Initialize blended latent representation as None.
    blended_latent_img = None

    for img, weight in zip(images, blending_weights):

        # Convert image to PyTorch tensor and normalize pixel values to [0, 1].
        scaled_image = torch.from_numpy(img).float() / 255.

        # Normalize and prepare image.
        permuted_image = (scaled_image * 2 - 1).permute(2, 0, 1).unsqueeze(0)

        # Encode image using VAE.
        latent_img = model.vae.encode(permuted_image.to(model.vae.device))['latent_dist'].mean * model.vae.config.scaling_factor

        # Blend the latent representation based on the weight.
        if blended_latent_img is None:
            blended_latent_img = latent_img * weight
        else:
            blended_latent_img += latent_img * weight

    # Reset VAE to Float16 if necessary.
    model.vae.to(dtype=torch.float16)

    # Return the blended latent representation.
    return blended_latent_img


### WEIGHTED SLERP (SPHERICAL LINEAR INTERPOLATION) ###
def weighted_slerp(weight, v0, v1):
    """Spherical linear interpolation with a weight factor."""
    v0_norm = v0 / torch.norm(v0, dim=-1, keepdim=True)
    v1_norm = v1 / torch.norm(v1, dim=-1, keepdim=True)
    dot_product = torch.sum(v0_norm * v1_norm, dim=-1, keepdim=True)
    omega = torch.acos(dot_product)
    sin_omega = torch.sin(omega)
    return (torch.sin((1.0 - weight) * omega) / sin_omega) * v0 + (torch.sin(weight * omega) / sin_omega) * v1

# def images_encoding_slerp(model, images: list[np.ndarray], blending_weights: list[float]):
#     """
#     Encode a list of images using the VAE model and blend their latent representations
#     using Weighted Spherical Interpolation (slerp) according to the given blending_weights.

#     Args:
#     - model: The StableDiffusionXLPipeline model.
#     - images: A list of numpy arrays, each representing an image.
#     - blending_weights: A list of floats representing the blending weights for each image.
#                         The blending_weights should sum to 1.

#     Returns:
#     - blended_latent_img: The blended latent representation.
#     """

#     # Ensure the blending_weights sum to 1.
#     assert len(images) == len(blending_weights), "The number of images and blending_weights must match."
#     assert np.isclose(sum(blending_weights), 1.0), "blending_weights must sum to 1."

#     # Set VAE to Float32 for encoding.
#     model.vae.to(dtype=torch.float32)

#     # Initialize variables to store valid latents and corresponding weights
#     valid_latents = []
#     valid_weights = []

#     # Iterate over images and weights
#     for idx, (img, weight) in enumerate(zip(images, blending_weights)):
#         if weight > 0.0:
#             # Convert image to PyTorch tensor and normalize pixel values to [0, 1].
#             scaled_image = torch.from_numpy(img).float() / 255.

#             # Normalize and prepare image.
#             permuted_image = (scaled_image * 2 - 1).permute(2, 0, 1).unsqueeze(0)

#             # Encode image using VAE.
#             latent_img = model.vae.encode(permuted_image.to(model.vae.device))['latent_dist'].mean * model.vae.config.scaling_factor

#             # Store valid latent and weight
#             valid_latents.append(latent_img)
#             valid_weights.append(weight)

#     # Convert valid_weights to tensor and normalize them
#     valid_weights = torch.tensor(valid_weights, device=model.vae.device, dtype=torch.float32)
#     valid_weights = valid_weights / valid_weights.sum()

#     # Perform SLERP only if there are valid latents
#     if len(valid_latents) == 1:
#         # If there's only one valid latent, no need to interpolate, just return it scaled by the weight
#         blended_latent_img = valid_latents[0] * valid_weights[0]
#     else:
#         # Perform SLERP for multiple valid latents
#         blended_latent_img = valid_latents[0] * valid_weights[0]
#         for i in range(1, len(valid_latents)):
#             blended_latent_img = weighted_slerp(valid_weights[i], blended_latent_img, valid_latents[i])

#     # Reset VAE to Float16 if necessary.
#     model.vae.to(dtype=torch.float16)

#     # Return the blended latent representation.
#     return blended_latent_img



# Function to dynamically update and apply the style arguments
def apply_style_aligned_args(handler, style_arg):
    handler.register(style_arg)



def images_encoding_slerp(model, images: list[np.ndarray], blending_weights: list[float], sa_args_list: list[StyleAlignedArgs], normal_famous_scaling: list[str], handler: Handler):
    """
    Encode a list of images using the VAE model and blend their latent representations
    using Weighted Spherical Interpolation (slerp) according to the given blending_weights.

    Args:
    - model: The StableDiffusionXLPipeline model.
    - images: A list of numpy arrays, each representing an image.
    - blending_weights: A list of floats representing the blending weights for each image.
                        The blending_weights should sum to 1.
    - sa_args_list: A list of StyleAlignedArgs for style alignment.
    - normal_famous_scaling: A list of classifications ("n" for normal, "f" for famous) for each image.

    Returns:
    - blended_latent_img: The blended latent representation.
    """

    # Ensure the blending_weights sum to 1.
    assert len(images) == len(blending_weights), "The number of images and blending_weights must match."
    assert np.isclose(sum(blending_weights), 1.0), "blending_weights must sum to 1."
    assert len(sa_args_list) == len(images), "You must provide exactly the same set of StyleAlignedArgs and number of images."
    assert len(normal_famous_scaling) == len(images), "The number of scaling classifications must match the number of images."

    # Set VAE to Float32 for encoding.
    model.vae.to(dtype=torch.float32)

    # Initialize variables to store valid latents and corresponding weights
    valid_latents = []
    valid_weights = []

    # Iterate over images and weights
    for idx, (img, weight, scaling_type) in enumerate(zip(images, blending_weights, normal_famous_scaling)):
        if weight > 0.0:
            # Determine which StyleAlignedArgs to use (based on scaling_type)
            style_args = sa_args_list[idx]  # Normal style or Famous style.

            print(f"Scaling Type: {scaling_type}")
            # Apply the style arguments dynamically
            if scaling_type == "n":
                apply_style_aligned_args(handler, style_args.normal_style)
            elif scaling_type == "f":
                apply_style_aligned_args(handler, style_args.famous_style)
            else:
                raise ValueError(f"Invalid scaling type: {scaling_type}")

            # Convert image to PyTorch tensor and normalize pixel values to [0, 1].
            scaled_image = torch.from_numpy(img).float() / 255.

            # Normalize and prepare image.
            permuted_image = (scaled_image * 2 - 1).permute(2, 0, 1).unsqueeze(0)

            # Encode image using VAE.
            latent_img = model.vae.encode(permuted_image.to(model.vae.device))['latent_dist'].mean * model.vae.config.scaling_factor

            # Store valid latent and weight
            valid_latents.append(latent_img)
            valid_weights.append(weight)

            # Unregister the handler after use (optional but recommended)
            handler.remove()

    # Convert valid_weights to tensor and normalize them
    valid_weights = torch.tensor(valid_weights, device=model.vae.device, dtype=torch.float32)
    valid_weights = valid_weights / valid_weights.sum()

    # Perform SLERP only if there are valid latents
    if len(valid_latents) == 1:
        # If there's only one valid latent, no need to interpolate, just return it scaled by the weight
        blended_latent_img = valid_latents[0] * valid_weights[0]
    else:
        # Perform SLERP for multiple valid latents
        blended_latent_img = valid_latents[0] * valid_weights[0]
        for i in range(1, len(valid_latents)):
            blended_latent_img = weighted_slerp(valid_weights[i], blended_latent_img, valid_latents[i])

    # Reset VAE to Float16 if necessary.
    model.vae.to(dtype=torch.float16)

    # Return the blended latent representation.
    return blended_latent_img
