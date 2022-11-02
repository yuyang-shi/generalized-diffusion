from dataclasses import dataclass
import math

import jax
import haiku as hk
import numpy as np
import jax.numpy as jnp

import haikumodels as hm
from .mlp import MLP


@dataclass
class Concat(hk.Module):
    def __init__(self, output_shape, hidden_shapes, act):
        super().__init__()
        self._layer = MLP(hidden_shapes=hidden_shapes, output_shape=output_shape, act=act)

    def __call__(self, x, t, context=None, is_training=True):
        if context is None:
            return self._layer(jnp.concatenate([x, t], axis=-1))
        else:
            return self._layer(jnp.concatenate([x, t, context], axis=-1))


@dataclass
class ConcatTimeEmbed(hk.Module):
    def __init__(self, output_shape, enc_shapes, t_dim, dec_shapes, act):
        super().__init__()
        self.temb_dim = t_dim
        t_enc_dim = t_dim * 2
        self.x_encoder = MLP(hidden_shapes=enc_shapes, output_shape=t_enc_dim, act=act)
        self.t_encoder = MLP(hidden_shapes=enc_shapes, output_shape=t_enc_dim, act=act)
        self.net = MLP(hidden_shapes=dec_shapes, output_shape=output_shape, act=act)

    def __call__(self, x, t, context=None, is_training=True):
        assert len(x.shape) == 2
        xemb = self.x_encoder(x)
        t = jnp.array(t, dtype=float).reshape(-1, 1)
        temb = get_timestep_embedding(t, self.temb_dim)
        temb = self.t_encoder(temb)
        temb = jnp.broadcast_to(temb, [xemb.shape[0], temb.shape[1]])
        return self.net(jnp.concatenate([xemb, temb], axis=-1))


# @dataclass
# class ConcatContextEmbed(hk.Module):
#     def __init__(self, output_shape, enc_shapes, c_dim, dec_shapes, act):
#         super().__init__()
#         self.net = MLP(hidden_shapes=dec_shapes, output_shape=output_shape, act=act)
#         self.c_encoder = MLP(hidden_shapes=enc_shapes, output_shape=c_dim, act=act)
# 
#     def __call__(self, x, t, context, is_training=True):
#         cemb = self.c_encoder(context)
#         h = jnp.concatenate([x, t, cemb], axis=-1)
#         out = self.net(h)
#         return out


@dataclass
class ConcatContextEmbed(hk.Module):
    def __init__(self, output_shape, enc_shapes, c_dim, dec_shapes, act):
        super().__init__()
        self.x_encoder = MLP(hidden_shapes=enc_shapes, output_shape=c_dim, act=act)
        self.c_encoder = MLP(hidden_shapes=enc_shapes, output_shape=c_dim, act=act)
        self.net = MLP(hidden_shapes=dec_shapes, output_shape=output_shape, act=act)

    def __call__(self, x, t, context, is_training=True):
        assert len(x.shape) == 2
        xemb = self.x_encoder(x)
        cemb = self.c_encoder(context)
        return self.net(jnp.concatenate([xemb, t, cemb], axis=-1))


@dataclass
class ConcatVisionContextEmbed(hk.Module):
    def __init__(self, output_shape, enc_shapes, c_dim, dec_shapes, act):
        super().__init__()
        self._vision_model = hm.ResNet50V2(include_top=False, weights="imagenet", pooling="avg")
        self.c_encoder = MLP(hidden_shapes=[], output_shape=c_dim, act="")
        self.x_encoder = MLP(hidden_shapes=enc_shapes, output_shape=c_dim, act=act)
        self.net = MLP(hidden_shapes=dec_shapes, output_shape=output_shape, act=act)

    def __call__(self, x, t, context, is_training=True):
        assert len(context.shape) == 4
        xemb = self.x_encoder(x)
        cemb = self._vision_model(context, is_training=is_training)
        cemb = self.c_encoder(cemb)
        return self.net(jnp.concatenate([xemb, t, cemb], axis=-1))


@dataclass
class SumVisionPositionContextTimeEmbed(hk.Module):
    def __init__(self, output_shape, enc_shapes, c_dim, dec_shapes, act):
        super().__init__()
        self.temb_dim = c_dim // 2
        self._vision_model = hm.ResNet50V2(include_top=False, weights="imagenet", pooling="avg")
        self.x_encoder = MLP(hidden_shapes=enc_shapes, output_shape=c_dim, act=act)
        self.t_encoder = MLP(hidden_shapes=enc_shapes, output_shape=c_dim, act=act)
        self.c_encoder = MLP(hidden_shapes=[], output_shape=c_dim, act="")
        self.net = MLP(hidden_shapes=dec_shapes, output_shape=output_shape, act=act)

    def __call__(self, x, t, context, is_training=True):
        assert len(x.shape) == 2
        assert len(context.shape) == 4

        xemb = get_positional_encoding(x, 6)
        xemb = self.x_encoder(xemb)

        temb = get_timestep_embedding(t, self.temb_dim)
        temb = self.t_encoder(temb)

        cemb = self._vision_model(context, is_training=is_training)
        cemb = self.c_encoder(cemb)
        return self.net(xemb + temb + cemb)


# @dataclass
# class ConcatVisionContextTimeEmbed(hk.Module):
#     def __init__(self, output_shape, enc_shapes, c_dim, dec_shapes, act):
#         super().__init__()
#         self.temb_dim = c_dim // 2
#         self._vision_model = hm.ResNet50V2(include_top=False, weights="imagenet", pooling="avg")
#         # self._vision_model = hk.nets.ResNet18(1000, resnet_v2=True)
#         self.c_encoder = MLP(hidden_shapes=enc_shapes, output_shape=c_dim, act="tanh")
#         self.x_encoder = MLP(hidden_shapes=enc_shapes, output_shape=c_dim, act=act)
#         self.t_encoder = MLP(hidden_shapes=enc_shapes, output_shape=c_dim, act="tanh")
#         self.net = MLP(hidden_shapes=dec_shapes, output_shape=output_shape, act=act)

#     def __call__(self, x, t, context, is_training=True):
#         if len(context.shape) != 4:
#             raise NotImplementedError
#         cemb = self._vision_model(context, is_training=is_training)
#         cemb = self.c_encoder(cemb)
#         xemb = self.x_encoder(x)
#         t = jnp.array(t, dtype=float).reshape(-1, 1)
#         temb = get_timestep_embedding(t, self.temb_dim)
#         temb = self.t_encoder(temb)
#         return self.net(jnp.concatenate([xemb, temb, cemb], axis=-1))


def get_timestep_embedding(timesteps, embedding_dim=128):
    """
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    https://github.com/pytorch/fairseq/blob/master/fairseq/modules/sinusoidal_positional_embedding.py
    """
    half_dim = embedding_dim // 2
    emb = math.log(10) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype=float) * -emb)

    emb = timesteps * jnp.expand_dims(emb, 0)
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], -1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = jnp.pad(emb, [0, 1])
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def get_positional_encoding(positions, embedding_dim=128):
    half_dim = embedding_dim // 2
    emb = 2. ** jnp.arange(half_dim, dtype=float)

    emb = jnp.expand_dims(positions, -1) * jnp.expand_dims(emb, (0, 1))
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], -1)
    emb = emb.reshape(emb.shape[0], -1)
    assert embedding_dim % 2 == 0
    assert emb.shape == (positions.shape[0], positions.shape[1] * embedding_dim)
    return emb
