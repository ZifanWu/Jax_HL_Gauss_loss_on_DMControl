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
        shift = jax.random.uniform(
            key,
            shape=(n, 1, 1, 2),
            minval=0,
            maxval=2 * pad + 1,
            dtype=jnp.float16
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
