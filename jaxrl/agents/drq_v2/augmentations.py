import jax
import jax.numpy as jnp
import flax.linen as nn
from jaxrl.networks.common import PRNGKey


def random_crop(key, img, padding):
    crop_from = jax.random.randint(key, (2, ), 0, 2 * padding + 1)
    crop_from = jnp.concatenate([crop_from, jnp.zeros((1, ), dtype=jnp.int32)])
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)),
                         mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


def batched_random_crop(key, imgs, padding=4):
    keys = jax.random.split(key, imgs.shape[0])
    return jax.vmap(random_crop, (0, 0, None))(keys, imgs, padding)


def drqv2_augmentation(x: jnp.ndarray, key: PRNGKey, pad=4) -> jnp.ndarray:
        # Create base grid
        n, h, w, c = x.shape
        eps = 1.0 / (h + 2 * pad)
        
        # pad the input
        padding = [(0, 0), (pad, pad), (pad, pad), (0, 0)]
        x_padded = jnp.pad(x, padding, mode='edge')
        arange = jnp.linspace(-1.0 + eps, 1.0 - eps, h + 2 * pad)[:h]
        arange_x = jnp.broadcast_to(arange[:, None], (h, h))
        arange_y = jnp.broadcast_to(arange[None, :], (h, h))
        base_grid = jnp.stack([arange_x, arange_y], axis=-1)
        base_grid = jnp.broadcast_to(base_grid[None, ...], (n, h, h, 2))
        
        # Generate random shifts
        shift = jax.random.randint(
            key,
            shape=(n, 1, 1, 2),
            minval=0,
            maxval=2 * pad + 1,
            dtype=x.dtype
        )
        shift = shift * (2.0 / (h + 2 * pad))
        
        # Apply shifts to grid
        grid = base_grid + shift
        
        # Convert grid from [-1, 1] to [0, H-1/W-1]
        h_padded, w_padded = x_padded.shape[1:3]
        grid = (grid + 1) / 2
        grid = grid * jnp.array([h_padded - 1, w_padded - 1])
        
        # Get corner points
        grid_i = jnp.floor(grid).astype(jnp.int32)
        grid_f = grid - grid_i
        
        # Clip coordinates
        grid_i = jnp.clip(grid_i, 0, jnp.array([h_padded - 1, w_padded - 1]))
        
        # Get corner indices
        i0, j0 = grid_i[..., 0], grid_i[..., 1]
        i1 = jnp.clip(i0 + 1, 0, h_padded - 1)
        j1 = jnp.clip(j0 + 1, 0, w_padded - 1)
        
        # Get weights
        wi0, wj0 = 1 - grid_f[..., 0], 1 - grid_f[..., 1]
        wi1, wj1 = grid_f[..., 0], grid_f[..., 1]
        
        # Gather and interpolate
        batch_idx = jnp.arange(n)[:, None, None]
        
        def gather_point(i, j):
            idx = (batch_idx, i, j)
            return x_padded[idx]
        
        return (wi0[..., None] * wj0[..., None] * gather_point(i0, j0) +
                wi1[..., None] * wj0[..., None] * gather_point(i1, j0) +
                wi0[..., None] * wj1[..., None] * gather_point(i0, j1) +
                wi1[..., None] * wj1[..., None] * gather_point(i1, j1))


# def drqv2_augmentation(x: jnp.ndarray, key: PRNGKey, pad=4) -> jnp.ndarray:
#     """
#     Reproduce the PyTorch RandomShiftsAug behavior in JAX
    
#     Args:
#         x (jnp.ndarray): Input tensor of shape (n, h, w, c)
#         key (jax.random.PRNGKey): Random number generation key
#         pad (int): Padding size for augmentation
    
#     Returns:
#         jnp.ndarray: Augmented tensor with random shifts
#     """
#     n, h, w, c = x.shape
#     assert h == w
    
#     # Pad the input
#     padding = ((0, 0), (pad, pad), (pad, pad), (0, 0))
#     x_padded = jnp.pad(x, padding, mode='edge')
    
#     # Create base grid
#     eps = 1.0 / (h + 2 * pad)
#     arange = jnp.linspace(-1.0 + eps, 1.0 - eps, h + 2 * pad)[:h]
    
#     # Create meshgrid
#     y, x_grid = jnp.meshgrid(arange, arange)
#     base_grid = jnp.stack([x_grid, y], axis=-1)
#     base_grid = base_grid.reshape(h, h, 2)
    
#     # Broadcast base grid to batch size
#     base_grid = jnp.broadcast_to(base_grid[None, ...], (n, h, h, 2))
    
#     # Generate random shifts
#     key, subkey = jax.random.split(key)
#     shift = jax.random.randint(
#         subkey, 
#         shape=(n, 1, 1, 2), 
#         minval=0, 
#         maxval=2 * pad + 1,
#         dtype=x.dtype
#     )
#     shift = shift * (2.0 / (h + 2 * pad))
    
#     # Apply shifts to grid
#     grid = base_grid + shift
    
#     def grid_sample(x, grid, padding_mode='zeros', align_corners=False):
#         """
#         Closely mimic PyTorch's grid_sample function in JAX.
        
#         Args:
#             x (jnp.ndarray): Input tensor of shape (N, C, H, W)
#             grid (jnp.ndarray): Sampling grid of shape (N, H_out, W_out, 2)
#             padding_mode (str): Padding mode ('zeros', 'border', or 'reflection')
#             align_corners (bool): If True, extrema (-1 and 1) are considered as referring 
#                                 to the center of the input's corner pixels
        
#         Returns:
#             jnp.ndarray: Sampled output tensor
#         """
#         # Transpose input to match PyTorch's NCHW format if needed
#         if x.ndim == 4 and x.shape[-1] < x.shape[1]:
#             x = x.transpose(0, 3, 1, 2)
        
#         C, H, W = x.shape
#         # _, H_out, W_out, _ = grid.shape
        
#         # Normalize grid coordinates
#         if not align_corners:
#             # PyTorch's align_corners=False scaling
#             grid = grid * jnp.array([(H-1)/2, (W-1)/2]) + jnp.array([(H-1)/2, (W-1)/2])
#         else:
#             # PyTorch's align_corners=True scaling
#             grid = (grid + 1) * jnp.array([(H-1)/2, (W-1)/2]) + jnp.array([(H-1)/2, (W-1)/2])
        
#         # Clip grid coordinates based on padding mode
#         def clip_coordinates(x, max_x):
#             return jnp.clip(x, 0, max_x - 1)
        
#         def reflect_coordinates(x, max_x):
#             # Reflect coordinates at the boundaries
#             x_reflected = jnp.abs(x)
#             x_reflected = jnp.where(
#                 x_reflected // (max_x - 1) % 2 == 1,
#                 (max_x - 1) - (x_reflected % (max_x - 1)),
#                 x_reflected % (max_x - 1)
#             )
#             return x_reflected
        
#         # Interpolation and sampling function
#         def bilinear_sample(img, x, y):
#             # Get grid coordinates
#             x0 = jnp.floor(x).astype(jnp.int32)
#             x1 = x0 + 1
#             y0 = jnp.floor(y).astype(jnp.int32)
#             y1 = y0 + 1
            
#             # Handle padding modes
#             if padding_mode == 'zeros':
#                 # Create masks for zero padding
#                 x0_valid = (x0 >= 0) & (x0 < W)
#                 x1_valid = (x1 >= 0) & (x1 < W)
#                 y0_valid = (y0 >= 0) & (y0 < H)
#                 y1_valid = (y1 >= 0) & (y1 < H)
                
#                 # Clip coordinates
#                 x0 = jnp.clip(x0, 0, W-1)
#                 x1 = jnp.clip(x1, 0, W-1)
#                 y0 = jnp.clip(y0, 0, H-1)
#                 y1 = jnp.clip(y1, 0, H-1)
#             elif padding_mode == 'border':
#                 x0 = clip_coordinates(x0, W)
#                 x1 = clip_coordinates(x1, W)
#                 y0 = clip_coordinates(y0, H)
#                 y1 = clip_coordinates(y1, H)
#             elif padding_mode == 'reflection':
#                 x0 = reflect_coordinates(x0, W).astype(jnp.int32)
#                 x1 = reflect_coordinates(x1, W).astype(jnp.int32)
#                 y0 = reflect_coordinates(y0, H).astype(jnp.int32)
#                 y1 = reflect_coordinates(y1, H).astype(jnp.int32)
#             else:
#                 raise ValueError(f"Unsupported padding mode: {padding_mode}")
            
#             # Get fractional parts
#             wx = x - x0
#             wy = y - y0
            
#             # Gather corner values
#             v00 = img[y0, x0]
#             v01 = img[y0, x1]
#             v10 = img[y1, x0]
#             v11 = img[y1, x1]
            
#             # Bilinear interpolation
#             v0 = v00 * (1 - wx) + v01 * wx
#             v1 = v10 * (1 - wx) + v11 * wx
            
#             # Final interpolation
#             v = v0 * (1 - wy) + v1 * wy
            
#             # For 'zeros' mode, apply zero padding mask
#             if padding_mode == 'zeros':
#                 valid_mask = (x0_valid & x1_valid & y0_valid & y1_valid)
#                 v = jnp.where(valid_mask[..., None], v, 0.0)
            
#             return v
        
#         # Process each channel
#         def sample_channel(channel):
#             def sample_batch_item(grid_item):
#                 return jax.vmap(bilinear_sample, in_axes=(None, 0, 0))(
#                     channel, grid_item[..., 0], grid_item[..., 1]
#                 )
#             return jax.vmap(sample_batch_item)(grid)
        
#         # Sample across all channels
#         output = jax.vmap(sample_channel)(x)
        
#         # Transpose back to input format if needed
#         return output.transpose(0, 2, 3, 1)
    
#     # Apply grid sampling to padded input
#     def sample_batch_item(batch_item, batch_grid):
#         return jax.vmap(grid_sample, in_axes=(0, None))(batch_item, batch_grid[0])

#     # Reshape input to match PyTorch's channel structure
#     x_reshaped = x_padded.reshape(n, -1, h + 2*pad, w + 2*pad, 3)
    
#     # Sample images
#     augmented = jax.vmap(sample_batch_item)(x_reshaped, grid)
    
#     # Reshape back to original shape
#     return augmented.reshape(n, h, w, c)