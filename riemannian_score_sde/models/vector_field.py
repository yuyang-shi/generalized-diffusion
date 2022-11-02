import abc

import numpy as np
import jax
import haiku as hk
import jax.numpy as jnp

from hydra.utils import instantiate
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.base import VectorSpace, EmbeddedManifold


class VectorFieldGenerator(hk.Module, abc.ABC):
    def __init__(self, architecture, embedding, output_shape, manifold):
        """X = fi * Xi with fi weights and Xi generators"""
        super().__init__()
        self.net = instantiate(architecture, output_shape=output_shape)
        self.manifold = manifold

    @staticmethod
    @abc.abstractmethod
    def output_shape(manifold):
        """Cardinality of the generating set."""

    def _weights(self, x, t, context=None, is_training=True):
        """shape=[..., card=n]"""
        return self.net(x, t, context=context, is_training=is_training)

    @abc.abstractmethod
    def _generators(self, x):
        """Set of generating vector fields: shape=[..., d, card=n]"""

    @property
    def decomposition(self):
        def weights_fn(x, t, context=None, is_training=True):
            return self._weights(x, t, context=context, is_training=is_training)
        return weights_fn, lambda x: self._generators(x)

    def __call__(self, x, t, context=None, is_training=True):
        fi_fn, Xi_fn = self.decomposition
        fi, Xi = fi_fn(x, t, context=context, is_training=is_training), Xi_fn(x)
        out = jnp.einsum("...n,...dn->...d", fi, Xi)
        # NOTE: seems that extra projection is required for generator=eigen
        # during the ODE solve cf tests/test_lkelihood.py
        out = self.manifold.to_tangent(out, x)
        return out

    def div_generators(self, x):
        """Divergence of the generating vector fields: shape=[..., card=n]"""


class DivFreeGenerator(VectorFieldGenerator):
    def __init__(self, architecture, embedding, output_shape, manifold):
        super().__init__(architecture, embedding, output_shape, manifold)

    @staticmethod
    def output_shape(manifold):
        return manifold.isom_group.dim

    def _generators(self, x):
        return self.manifold.div_free_generators(x)

    def div_generators(self, x):
        shape = [*x.shape[:-1], self.output_shape(self.manifold)]
        return jnp.zeros(shape)


class EigenGenerator(VectorFieldGenerator):
    """Gradient of laplacien eigenfunctions with eigenvalue=1"""

    def __init__(self, architecture, embedding, output_shape, manifold):
        super().__init__(architecture, embedding, output_shape, manifold)
        assert isinstance(manifold, Hypersphere)

    @staticmethod
    def output_shape(manifold):
        return manifold.embedding_space.dim

    def _generators(self, x):
        return self.manifold.eigen_generators(x)

    def div_generators(self, x):
        # NOTE: Empirically need this factor 2 to match AmbientGenerator but why??
        return -self.manifold.dim * 2 * x


class AmbientGenerator(VectorFieldGenerator):
    """Equivalent to EigenGenerator"""

    def __init__(self, architecture, embedding, output_shape, manifold):
        super().__init__(architecture, embedding, output_shape, manifold)

    @staticmethod
    def output_shape(manifold):
        if isinstance(manifold, EmbeddedManifold):
            output_shape = manifold.embedding_space.dim
        else:
            output_shape = manifold.dim
        return output_shape

    def _generators(self, x):
        return self.manifold.eigen_generators(x)

    def __call__(self, x, t, context=None, is_training=True):
        # `to_tangent`` have an 1/sq_norm(x) term that wrongs the div
        return self.manifold.to_tangent(self.net(x, t, context=context, is_training=is_training), x)


class LieAlgebraGenerator(VectorFieldGenerator):
    def __init__(self, architecture, embedding, output_shape, manifold):
        super().__init__(architecture, embedding, output_shape, manifold)

    @staticmethod
    def output_shape(manifold):
        return manifold.dim

    def _generators(self, x):
        return self.manifold.lie_algebra.basis

    def __call__(self, x, t, context=None, is_training=True):
        x = x.reshape((x.shape[0], self.manifold.dim, self.manifold.dim))
        fi_fn, Xi_fn = self.decomposition
        x_input = x.reshape((*x.shape[:-2], -1))
        # x_input = self.manifold.vee(self.manifold.log(x)) #NOTE: extremely unstable
        fi, Xi = fi_fn(x_input, t, context=context, is_training=is_training), Xi_fn(x)
        out = jnp.einsum("...i,ijk ->...jk", fi, Xi)
        # is_tangent = self.manifold.lie_algebra.belongs(out, atol=1e-3).all()
        out = self.manifold.compose(x, out)
        # is_tangent = self.manifold.is_tangent(out, x, atol=1e-3).all()
        # out = self.manifold.to_tangent(out, x)
        return out.reshape((x.shape[0], -1))


class TorusGenerator(VectorFieldGenerator):
    def __init__(self, architecture, embedding, output_shape, manifold):
        super().__init__(architecture, embedding, output_shape, manifold)

        self.rot_mat = jnp.array([[0, -1], [1, 0]])

    @staticmethod
    def output_shape(manifold):
        return manifold.dim

    def _generators(self, x):
        return (
            self.rot_mat @ x.reshape((*x.shape[:-1], self.manifold.dim, 2))[..., None]
        )[..., 0]

    def __call__(self, x, t, context=None, is_training=True):
        weights_fn, fields_fn = self.decomposition
        weights = weights_fn(x, t, context=context, is_training=is_training)
        fields = fields_fn(x)

        return (fields * weights[..., None]).reshape(
            (*x.shape[:-1], self.manifold.dim * 2)
        )
