"""Modified code from https://github.com/yang-song/score_sde"""
# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All functions related to loss computation and optimization.
"""

from typing import Callable, Tuple

import jax
import optax
import jax.numpy as jnp
import jax.random as random

from score_sde.utils import batch_mul
from score_sde.models import get_score_fn, PushForward, SDEPushForward
from score_sde.utils import ParametrisedScoreFunction, TrainState
from score_sde.models import div_noise, get_div_fn
from score_sde.utils import get_exact_jac_fn


def get_dsm_loss_fn(
    pushforward: SDEPushForward,
    model: ParametrisedScoreFunction,
    train: bool = True,
    reduce_mean: bool = True,
    like_w: bool = True,
    eps: float = 1e-3,
):
    sde = pushforward.sde
    reduce_op = (
        jnp.mean
        if reduce_mean
        else lambda *args, **kwargs: 0.5 * jnp.sum(*args, **kwargs)
    )

    def loss_fn(
        rng: jax.random.KeyArray, params: dict, states: dict, batch: dict
    ) -> Tuple[float, dict]:
        score_fn = get_score_fn(
            sde,
            model,
            params,
            states,
            train=train,
            return_state=True,
        )
        x_0, context = pushforward.transform.inv(batch["data"]), batch["context"]

        rng, step_rng = random.split(rng)
        # uniformly sample from SDE timeframe
        t = random.uniform(step_rng, (x_0.shape[0],), minval=sde.t0 + eps, maxval=sde.tf)
        rng, step_rng = random.split(rng)
        z = random.normal(step_rng, x_0.shape)
        mean, std = sde.marginal_prob(x_0, t)
        # reparametrised sample x_t|x_0 = mean + std * z with z ~ N(0,1)
        x_t = mean + batch_mul(std, z)
        score, new_model_state = score_fn(x_t, t, context, rng=step_rng)
        # grad log p(x_t|x_0) = - 1/std^2 (x_t - mean) = - z / std

        if not like_w:
            losses = jnp.square(batch_mul(score, std) + z)
            # losses = std^2 * DSM(x_t, x_0)
            losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1)
        else:  # maximum likelihood training
            g2 = sde.coefficients(jnp.zeros_like(x_0), t)[1] ** 2
            # losses = DSM(x_t, x_0)
            losses = jnp.square(score + batch_mul(z, 1.0 / std))
            losses = reduce_op(losses.reshape((losses.shape[0], -1)), axis=-1) * g2

        loss = jnp.mean(losses)
        return loss, new_model_state

    return loss_fn


def get_ism_loss_fn(
    pushforward: SDEPushForward,
    model: ParametrisedScoreFunction,
    train: bool,
    reduce_mean: bool = True,
    like_w: bool = True,
    hutchinson_type="Rademacher",
    eps: float = 1e-3,
):
    sde = pushforward.sde

    def loss_fn(
        rng: jax.random.KeyArray, params: dict, states: dict, batch: dict
    ) -> Tuple[float, dict]:
        score_fn = get_score_fn(
            sde,
            model,
            params,
            states,
            train=train,
            return_state=True,
        )
        x_0, context = pushforward.transform.inv(batch["data"]), batch["context"]

        rng, step_rng = random.split(rng)
        t = random.uniform(step_rng, (x_0.shape[0],), minval=sde.t0 + eps, maxval=sde.tf)

        rng, step_rng = random.split(rng)
        x_t = sde.marginal_sample(step_rng, x_0, t)
        score, new_model_state = score_fn(x_t, t, context, rng=step_rng)

        # ISM loss
        rng, step_rng = random.split(rng)
        epsilon = div_noise(step_rng, x_0.shape, hutchinson_type)
        drift_fn = lambda x, t, context: score_fn(x, t, context, rng=step_rng)[0]
        div_fn = get_div_fn(drift_fn, hutchinson_type)
        div_score = div_fn(x_t, t, context, epsilon)
        sq_norm_score = jnp.power(score, 2).sum(axis=-1)
        losses = 0.5 * sq_norm_score + div_score

        if like_w:
            g2 = sde.coefficients(jnp.zeros_like(x_0), t)[1] ** 2
            losses = losses * g2

        assert len(losses.shape) == 1
        loss = jnp.mean(losses)
        
        if reduce_mean:
            loss = loss / x_0.shape[-1]
     
        return loss, new_model_state

    return loss_fn


def get_ism_simplex_loss_fn(
    pushforward: SDEPushForward,
    model: ParametrisedScoreFunction,
    train: bool,
    reduce_mean: bool = True,
    like_w: bool = True,
    hutchinson_type="Rademacher",
    eps: float = 1e-3,
    fast_sampling=True, 
):
    sde = pushforward.sde
    if hutchinson_type == "None":
        print("Using exact jacobian")
    else:
        print("Using Gaussian noise for estimating divergence")

    def loss_fn(
        rng: jax.random.KeyArray, params: dict, states: dict, batch: dict
    ) -> Tuple[float, dict]:
        score_fn_raw = get_score_fn(
            sde,
            model,
            params,
            states,
            train=train,
            return_state=True,
            std_trick=False
        )
        if sde.inv_scale:
            def score_fn(x, t, context, rng):
                score_raw, new_model_state = score_fn_raw(x, t, context, rng=rng)
                return score_raw / x, new_model_state
                # return (sde.alpha - 1 - score_raw) / x, new_model_state
        else:
            score_fn = score_fn_raw
        x_0, context = pushforward.transform.inv(batch["data"]), batch["context"]

        rng, step_rng = random.split(rng)
        t = random.uniform(step_rng, (x_0.shape[0],), minval=sde.t0 + eps, maxval=sde.tf)

        rng, step_rng = random.split(rng)
        x_t = sde.marginal_sample(step_rng, x_0, t, fast_sampling=fast_sampling)
        score, new_model_state = score_fn(x_t, t, context, rng=step_rng)
        D = x_t.shape[-1]

        # ISM loss
        rng, step_rng = random.split(rng)
        drift_fn = lambda x, t, context: score_fn(x, t, context, rng=step_rng)[0]

        if hutchinson_type == "None":
            jac_fn = get_exact_jac_fn(drift_fn)
            jac_score = jac_fn(x_t, t, context)
            outer_score = jnp.expand_dims(score, axis=-1) * jnp.expand_dims(score, axis=-2)
            losses = (((jnp.eye(D) - jnp.expand_dims(x_t, axis=-2)) * (jac_score + 0.5 * outer_score)).sum(axis=-1) * x_t).sum(axis=-1)
            losses = losses + ((1 - D*x_t) * score).sum(axis=-1)

        elif hutchinson_type == "Gaussian":
            wf_cov_decomp = sde.wf_cov_decomp(x_t)
            rng, step_rng = random.split(rng)
            epsilon = jnp.einsum("...ij,...j->...i", wf_cov_decomp, jax.random.normal(step_rng, x_0.shape))
            div_fn = get_div_fn(drift_fn, "Gaussian")
            div_score = div_fn(x_t, t, context, epsilon)

            losses = div_score + 0.5 * (jnp.sum(x_t * score**2, axis=-1) - jnp.sum(x_t * score, axis=-1)**2) + jnp.sum((1 - D*x_t) * score, axis=-1)
        
        else:
            raise NotImplementedError

        losses = losses - D * (D-1) / 2 + jnp.sum(sde.alpha) * (D-1) / 2

        if like_w:
            g2 = sde.beta_t(t)
            losses = losses * g2

        assert len(losses.shape) == 1
        loss = jnp.mean(losses)

        if reduce_mean:
            loss = loss / D
        
        return loss, new_model_state

    return loss_fn


def get_logp_loss_fn(
    pushforward: PushForward,
    model: ParametrisedScoreFunction,
    train: bool = True,
    **kwargs
):
    def loss_fn(
        rng: jax.random.KeyArray, params: dict, states: dict, batch: dict
    ) -> Tuple[float, dict]:
        x_0 = batch["data"]

        model_w_dicts = (model, params, states)
        log_prob = pushforward.get_log_prob(model_w_dicts, train=train)
        losses = -log_prob(x_0, rng=rng)[0]
        loss = jnp.mean(losses)

        return loss, states

    return loss_fn


def get_ema_loss_step_fn(
    loss_fn,
    optimizer,
    train: bool,
):
    """Create a one-step training/evaluation function.

    Args:
      loss_fn: loss function to compute
      train: `True` for training and `False` for evaluation.

    Returns:
      A one-step function for training or evaluation.
    """

    def step_fn(carry_state: Tuple[jax.random.KeyArray, TrainState], batch: dict):
        """Running one step of training or evaluation.

        This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
        for faster execution.

        Args:
          carry_state: A tuple (JAX random state, NamedTuple containing the training state).
          batch: A mini-batch of training/evaluation data.

        Returns:
          new_carry_state: The updated tuple of `carry_state`.
          loss: The average loss value of this state.
        """

        (rng, train_state) = carry_state
        rng, step_rng = jax.random.split(rng)
        grad_fn = jax.value_and_grad(loss_fn, argnums=1, has_aux=True)
        if train:
            params = train_state.params
            model_state = train_state.model_state
            (loss, new_model_state), grad = grad_fn(step_rng, params, model_state, batch)
            updates, new_opt_state = optimizer.update(grad, train_state.opt_state)
            new_parmas = optax.apply_updates(params, updates)

            new_params_ema = jax.tree_map(
                lambda p_ema, p: p_ema * train_state.ema_rate
                + p * (1.0 - train_state.ema_rate),
                train_state.params_ema,
                new_parmas,
            )
            step = train_state.step + 1
            new_train_state = train_state._replace(
                step=step,
                opt_state=new_opt_state,
                model_state=new_model_state,
                params=new_parmas,
                params_ema=new_params_ema,
            )
        else:
            loss, _ = loss_fn(
                step_rng, train_state.params_ema, train_state.model_state, batch
            )
            new_train_state = train_state

        new_carry_state = (rng, new_train_state)
        return new_carry_state, loss

    return step_fn
