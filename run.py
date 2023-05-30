import os
import socket
import logging
from timeit import default_timer as timer
from tqdm import tqdm

import numpy as np
import jax
from jax import numpy as jnp
import optax
import haiku as hk

from omegaconf import OmegaConf
from hydra.utils import instantiate, get_class, call

from score_sde.models.flow import SDEPushForward
from score_sde.losses import get_ema_loss_step_fn
from score_sde.utils import TrainState, save, restore
from score_sde.utils.loggers_pl import LoggerCollection
from score_sde.datasets import random_split, DataLoader, TensorDataset
from riemannian_score_sde.utils.normalization import compute_normalization
from riemannian_score_sde.utils.vis import plot, plot_ref

log = logging.getLogger(__name__)


def run(cfg):
    def train(train_state):
        loss_fn = instantiate(
            cfg.loss, pushforward=pushforward, model=model, eps=cfg.eps, train=True
        )
        train_step_fn = get_ema_loss_step_fn(loss_fn, optimizer=optimiser, train=True)
        train_step_fn = jax.jit(train_step_fn)

        rng = train_state.rng
        t = tqdm(
            range(train_state.step, cfg.steps),
            total=cfg.steps - train_state.step,
            bar_format="{desc}{bar}{r_bar}",
            mininterval=3,
        )
        train_time = timer()
        total_train_time = 0
        for step in t:
            data, context = next(train_ds)
            data = jnp.repeat(data, cfg.num_repeat_data, 0)
            batch = {"data": data, "context": context}
            rng, next_rng = jax.random.split(rng)
            (rng, train_state), loss = train_step_fn((next_rng, train_state), batch)
            if jnp.isnan(loss).any():
                log.warning("Loss is nan")
                return train_state, False

            if step % 50 == 0:
                logger.log_metrics({"train/loss": loss}, step)
                t.set_description(f"Loss: {loss:.3f}")

            if step == 0:
                if cfg.train_plot:
                    generate_plots(train_state, "val", step=step, forward_only=True)
            elif (step + 1) % cfg.val_freq == 0:
                logger.log_metrics(
                    {"train/time_per_it": (timer() - train_time) / cfg.val_freq}, step
                )
                total_train_time += timer() - train_time
                save(ckpt_path, train_state)
                eval_time = timer()
                if cfg.train_val:
                    evaluate(train_state, "val", step)
                    logger.log_metrics({"val/time_per_it": (timer() - eval_time)}, step)
                if cfg.train_plot:
                    generate_plots(train_state, "val", step=step)
                train_time = timer()
            
            if step == cfg.steps - 1:
                save(ckpt_final_path, train_state)

        logger.log_metrics({"train/total_time": total_train_time}, step)
        return train_state, True

    def evaluate(train_state, stage, step=None):
        log.info("Running evaluation")
        dataset = eval_ds if stage == "val" else test_ds

        model_w_dicts = (model, train_state.params_ema, train_state.model_state)
        likelihood_fn = pushforward.get_log_prob(model_w_dicts, train=False)

        logp, nfe, N = 0.0, 0.0, 0

        if hasattr(dataset, "__len__"):
            for batch in dataset:
                if cfg.test_ode:
                    logp_step, nfe_step = likelihood_fn(*batch)
                    logp += logp_step.sum()
                    nfe += nfe_step
                N += batch[0].shape[0]
        else:
            dataset.batch_dims = [cfg.eval_batch_size]
            samples = round(cfg.eval_num_data / cfg.eval_batch_size)
            for i in range(samples):
                batch = next(dataset)
                if cfg.test_ode:
                    logp_step, nfe_step = likelihood_fn(*batch)
                    logp += logp_step.sum()
                    nfe += nfe_step
                N += batch[0].shape[0]
            dataset.batch_dims = cfg.dataset.batch_dims

        if cfg.test_ode:
            logp /= N
            nfe /= len(dataset) if hasattr(dataset, "__len__") else samples

            logger.log_metrics({f"{stage}/logp": logp}, step)
            log.info(f"{stage}/logp = {logp:.3f}")
            logger.log_metrics({f"{stage}/nfe": nfe}, step)
            log.info(f"{stage}/nfe = {nfe:.1f}")

            if stage == "test":  # Estimate normalisation constant
                default_context = context[0] if context is not None else None
                Z = compute_normalization(
                    likelihood_fn, data_manifold, context=default_context
                )
                log.info(f"Z = {Z:.2f}")
                logger.log_metrics({f"{stage}/Z": Z}, step)


    def generate_plots(train_state, stage, step=None, forward_only=False):
        log.info("Generating plots")
        rng = jax.random.PRNGKey(cfg.seed)
        dataset = eval_ds if stage == "val" else test_ds

        model_w_dicts = (model, train_state.params_ema, train_state.model_state)
        sampler_kwargs = dict(N=cfg.sampler_N, eps=cfg.eps, predictor="GRW")

        M = cfg.plot_M
        x0, context = next(dataset)
        shape = (int(x0.shape[0] * M), *transform.inv(x0).shape[1:])
        if context is not None:
            x0 = jnp.repeat(x0, M, 0)
            context = jnp.repeat(context, M, 0)
        else:
            for _ in range(M - 1):
                x0 = jnp.concatenate([x0, next(dataset)[0]])
        
        if not forward_only:
            ## p_0 (backward)
            sampler = pushforward.get_sampler(model_w_dicts, train=False, **sampler_kwargs)
            rng, next_rng = jax.random.split(rng)
            x = sampler(next_rng, shape, context)
            prop_in_M = data_manifold.belongs(x, atol=1e-4).mean()
            log.info(f"Prop samples in M: {100 * prop_in_M.item()}")
            plt = plot(data_manifold, [x0], [x], dataset=dataset)
            logger.log_plot(f"{stage}_x0_backw", plt, step if step is not None else cfg.steps)
            del sampler

        ## p_T (forward)
        if isinstance(pushforward, SDEPushForward):
            sampler = pushforward.get_sampler(
                model_w_dicts, train=False, reverse=False, **sampler_kwargs
            )
            z = transform.inv(x0)
            rng, next_rng = jax.random.split(rng)
            zT = sampler(next_rng, None, context, z=z)
            plt = plot_ref(model_manifold, transform.inv(zT))
            if plt is not None:
                logger.log_plot(f"{stage}_xT", plt, step if step is not None else cfg.steps)
            del sampler
        logger.save()

    ### Main
    log.info("Stage : Startup")
    log.info(f"Jax devices: {jax.devices()}")
    run_path = os.getcwd()
    log.info(f"run_path: {run_path}")
    log.info(f"hostname: {socket.gethostname()}")
    ckpt_path = os.path.join(run_path, cfg.ckpt_dir)
    ckpt_final_path = os.path.join(run_path, cfg.ckpt_dir + '_final')
    os.makedirs(ckpt_path, exist_ok=cfg.mode == "test")
    os.makedirs(ckpt_final_path, exist_ok=cfg.mode == "test")
    loggers = [instantiate(logger_cfg) for logger_cfg in cfg.logger.values()]
    logger = LoggerCollection(loggers)
    logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    log.info("Stage : Instantiate model")
    rng = jax.random.PRNGKey(cfg.seed)
    data_manifold = instantiate(cfg.manifold)
    transform = instantiate(cfg.transform, data_manifold)
    model_manifold = transform.domain
    flow = instantiate(cfg.flow, manifold=model_manifold)
    base = instantiate(cfg.base, model_manifold, flow)
    pushforward = instantiate(cfg.pushf, flow, base, transform=transform)

    log.info("Stage : Instantiate dataset")
    rng, next_rng = jax.random.split(rng)
    dataset = instantiate(cfg.dataset, rng=next_rng)

    if isinstance(dataset, TensorDataset):
        # split and wrapp dataset into dataloaders
        train_ds, eval_ds, test_ds = random_split(
            dataset, lengths=cfg.splits, rng=next_rng
        )
        train_ds, eval_ds, test_ds = (
            DataLoader(train_ds, batch_dims=cfg.batch_size, rng=next_rng, shuffle=True),
            DataLoader(eval_ds, batch_dims=cfg.eval_batch_size, rng=next_rng),
            DataLoader(test_ds, batch_dims=cfg.eval_batch_size, rng=next_rng),
        )
        log.info(
            f"Train size: {len(train_ds.dataset)}. Val size: {len(eval_ds.dataset)}. Test size: {len(test_ds.dataset)}"
        )
    else:
        train_ds, eval_ds, test_ds = dataset, dataset, dataset

    log.info("Stage : Instantiate vector field model")

    def model(y, t, context=None, is_training=True):
        """Vector field s_\theta: y, t, context -> T_y M"""
        output_shape = get_class(cfg.generator._target_).output_shape(model_manifold)
        score = instantiate(
            cfg.generator,
            cfg.architecture,
            cfg.embedding,
            output_shape,
            manifold=model_manifold,
        )
        # TODO: parse context into embedding map
        
        t = jnp.expand_dims(t.reshape(-1), -1)
        if context is not None:
            # if context.shape[0] != y.shape[0]:
            #     raise ValueError
            #     context = jnp.repeat(jnp.expand_dims(context, 0), y.shape[0], 0)
            return score(y, t, context=context, is_training=is_training)
        else:
            return score(y, t, is_training=is_training)

    model = hk.transform_with_state(model)

    rng, next_rng = jax.random.split(rng)
    t = jnp.zeros((cfg.batch_size * cfg.num_repeat_data, 1))
    data, context = next(train_ds)
    data = jnp.repeat(data, cfg.num_repeat_data, 0)
    params, state = model.init(rng=next_rng, y=transform.inv(data), t=t, context=context)

    log.info("Stage : Instantiate optimiser")
    schedule_fn = instantiate(cfg.scheduler)
    optimiser = optax.chain(instantiate(cfg.optim), optax.scale_by_schedule(schedule_fn))
    opt_state = optimiser.init(params)

    if cfg.resume or cfg.mode == "test":  # if resume or evaluate
        train_state = restore(ckpt_path)
    else:
        rng, next_rng = jax.random.split(rng)
        train_state = TrainState(
            opt_state=opt_state,
            model_state=state,
            step=0,
            params=params,
            ema_rate=cfg.ema_rate,
            params_ema=params,
            rng=next_rng,  # TODO: we should actually use this for reproducibility
        )
        save(ckpt_path, train_state)

    if cfg.mode == "train" or cfg.mode == "all":
        log.info("Stage : Training")
        train_state, success = train(train_state)
    if cfg.mode == "test" or (cfg.mode == "all" and success):
        log.info("Stage : Test")
        if cfg.test_val:
            evaluate(train_state, "val")
        if cfg.test_test:
            evaluate(train_state, "test")
        if cfg.test_plot:
            generate_plots(train_state, "test")
        success = True
    logger.save()
    logger.finalize("success" if success else "failure")
