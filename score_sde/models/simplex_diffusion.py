import jax
import jax.numpy as jnp


def β(θ, t): 
    # return jnp.where(θ==0, θ, 1/2*(θ-1)*t)
    return 1/2*(θ-1)*t

def η_of_β(β):
    # return jnp.where(jnp.logical_or(β==0, β==1), β, β/(jnp.exp(β)-1))
    return jnp.where(β==0, jnp.ones_like(β), β/(jnp.exp(β)-1))

η = lambda θ, t: η_of_β(β(θ, t))

μ = lambda t, θ: 2*η(θ, t)/t

def σ_squared(t, θ):
    β_θ_t = β(θ, t)
    η_θ_t = η_of_β(β_θ_t)
    return jnp.where(θ == 1, 2/(3*t), 2*η_θ_t/t*(η_θ_t+β_θ_t)**2*(1+η_θ_t/(η_θ_t+β_θ_t)-2*η_θ_t)/β_θ_t**2)

"""
      Normal approximation of the transition function, valid for small time steps. In practice, used when t<=0.05. This is Theorem 1 of Jenkins, P. A., & Spano, D. (2017). Exact simulation of the Wright--Fisher diffusion. The Annals of Applied Probability, 27(3), 1478–1509.
"""
def Compute_A_approx(rng, θ, t):
  A_real = jax.random.normal(rng, shape=t.shape) * jnp.sqrt(σ_squared(t, θ)) + μ(t, θ)
  return jnp.where(A_real<0, 0, jnp.rint(A_real))


def log_akmθ(θ, k, m):  # k >= m
    from jax.scipy.special import gammaln
    # return jnp.where(k==0, 0, jnp.log(θ+2*k-1) + gammaln(θ+m+k-1) - gammaln(θ+m) - gammaln(m+1) - gammaln(k-m+1))
    return jnp.log(θ+2*k-1) + gammaln(θ+m+k-1) - gammaln(θ+m) - gammaln(m+1) - gammaln(k-m+1)

def log_bk_t_θ_t(k, t, θ, m):
    return log_akmθ(θ, k, m) - k*(k+θ-1)*t/2

def C_m_t_θ(m, t, θ):
    def cond_fun(val):
        log_bkm_current, log_bkm_p1, ind = val
        return log_bkm_p1 >= log_bkm_current

    def body_fun(val):
        log_bkm_current, log_bkm_p1, ind = val
        ind = ind + 1
        log_bkm_current = log_bkm_p1
        log_bkm_p1 = log_bk_t_θ_t(m + ind + 1, t, θ, m)
        return log_bkm_current, log_bkm_p1, ind

    log_bkm_current = log_bk_t_θ_t(m, t, θ, m)
    log_bkm_p1 = log_bk_t_θ_t(m + 1, t, θ, m)
    log_bkm_current, log_bkm_p1, ind = jax.lax.while_loop(cond_fun, body_fun, (log_bkm_current, log_bkm_p1, jnp.zeros_like(t)))
    return ind


def Compute_A_good_start_arb(rng, θ, t):
    u = jax.random.uniform(rng, shape=t.shape)
    # m0 = jnp.rint(μ(t, θ))
    m0 = jnp.zeros_like(t)
    return jax.vmap(Compute_A_good_m_start_arb)(u, θ, t, m0)


def signed_logaddexp(log_x1, log_x2, sign2, sign1=1.):
    logadd_x1x2 = jnp.logaddexp(log_x1, log_x2)
    # log(abs(x1 - x2))
    log_abs_minus_x1x2 = jnp.where(log_x1 > log_x2, jnp.log(1-jnp.exp(log_x2 - log_x1)) + log_x1,  # log (x1 - x2)
                                                    jnp.log(1-jnp.exp(log_x1 - log_x2)) + log_x2)  # log (x2 - x1)    
    return jnp.where(sign1 == sign2, logadd_x1x2, log_abs_minus_x1x2), jnp.where(sign1 == sign2, sign1, jnp.where(log_x1 > log_x2, 1., -1.) * jnp.where(sign1 == 1, 1., -1.))

def minus_exp(val_pair):
    logsum, sign = val_pair
    return sign * jnp.exp(logsum)

MAX_KLEN = 100  # maximum length of k, assume simulated M never exceed this number
def Compute_A_good_m_start_arb(U, θ, t, m0):
    def cond_fun_inner(val):
        kvec, m, S_kvec_M_BOTH = val
        return jnp.logical_and(minus_exp(S_kvec_M_BOTH[0]) < U, minus_exp(S_kvec_M_BOTH[1]) > U)

    def body_fun_inner(val):
        kvec, m, S_kvec_M_BOTH = val
        kvec = kvec + 1
        S_kvec_M_BOTH = S_kvec_M_both_logsumexp_addk(kvec, t, θ, m, S_kvec_M_BOTH)
        return kvec, m, S_kvec_M_BOTH

    def cond_fun_outer(val):
        kvec, m, S_kvec_M_BOTH = val
        return minus_exp(S_kvec_M_BOTH[0]) < U  # S_kvec_M_minus < U

    def body_fun_outer(val):
        kvec, m, S_kvec_M_BOTH = val
        m = m + 1
        # kvec = jnp.where(jnp.arange(MAX_KLEN + 1)==m, jnp.ceil(C_m_t_θ(m, t, θ) / 2), kvec)
        kvec = jnp.where(jnp.arange(MAX_KLEN + 1)==m, kvec0, kvec)
        S_kvec_M_BOTH = S_kvec_M_both_logsumexp_addM(kvec, t, θ, m, S_kvec_M_BOTH)
        kvec, m, S_kvec_M_BOTH = jax.lax.while_loop(cond_fun_inner, body_fun_inner, (kvec, m, S_kvec_M_BOTH))  # Tune kvec
        return kvec, m, S_kvec_M_BOTH
  
    # kvec0 = jnp.zeros(MAX_KLEN + 1)
    # kvec0 = jnp.where(jnp.arange(MAX_KLEN + 1)<=m0, jnp.ceil(jax.vmap(C_m_t_θ, in_axes=(0, None, None))(jnp.arange(MAX_KLEN + 1), t, θ) / 2), kvec0)
    kvec0 = jnp.ceil(jax.vmap(C_m_t_θ, in_axes=(0, None, None))(jnp.arange(MAX_KLEN + 1), t, θ) / 2)
    S_kvec_M_BOTH = S_kvec_M_both_logsumexp_arb(kvec0, t, θ, m0)
    kvec, m, S_kvec_M_BOTH = jax.lax.while_loop(cond_fun_inner, body_fun_inner, (kvec0, m0, S_kvec_M_BOTH))  # Tune kvec
    kvec, m, S_kvec_M_BOTH = jax.lax.while_loop(cond_fun_outer, body_fun_outer, (kvec, m, S_kvec_M_BOTH))
    return m

def S_kvec_M_both_logsumexp_arb(kvec, t, θ, M):  # Compute arbitrary double sum given kvec, t, θ, M from scratch
    two_kvec_plus_1 = 2*kvec + 1

    def body_fun_inner(i, val):
        logsum, sign, m = val
        sign_b = jnp.where(i%2==0, 1, -1)
        log_b = log_bk_t_θ_t(m+i, t, θ, m)
        logsum, sign = signed_logaddexp(logsum, log_b, sign1=sign, sign2=sign_b)
        return logsum, sign, m

    # Compute log_S_kvec_M_minus
    def body_fun_outer_1(m, val):
        logsum, sign = val
        two_km_plus_1 = two_kvec_plus_1[..., m]
        logsum, sign, m = jax.lax.fori_loop(0, two_km_plus_1.astype(int) + 1, body_fun_inner, (logsum, sign, m))
        return logsum, sign

    log_S_kvec_M_minus, log_S_kvec_M_minus_sign = jax.lax.fori_loop(0, M.astype(int) + 1, body_fun_outer_1, (-jnp.inf * jnp.ones_like(t), jnp.ones_like(t)))

    def body_fun_outer_2(m, val):
        logsum_m = val
        two_km_plus_1 = two_kvec_plus_1[..., m]
        log_b = log_bk_t_θ_t(m+two_km_plus_1, t, θ, m)
        logsum_m = jnp.logaddexp(logsum_m, log_b)
        return logsum_m

    logsum_m_minus = jax.lax.fori_loop(0, M.astype(int) + 1, body_fun_outer_2, -jnp.inf * jnp.ones_like(t))
    log_S_kvec_M_plus, log_S_kvec_M_plus_sign = signed_logaddexp(log_S_kvec_M_minus, logsum_m_minus, sign1=log_S_kvec_M_minus_sign, sign2=1)

    return (log_S_kvec_M_minus, log_S_kvec_M_minus_sign), (log_S_kvec_M_plus, log_S_kvec_M_plus_sign)

def S_kvec_M_both_logsumexp_addM(kvec, t, θ, M, S_kvec_M_BOTH):
    # km = jnp.ceil(C_m_t_θ(M, t, θ) / 2)
    km = kvec[M.astype(int)]
    two_km_plus_1 = 2 * km + 1
    
    log_S_kvec_M_minus, log_S_kvec_M_minus_sign = S_kvec_M_BOTH[0]
    log_S_kvec_M_plus, log_S_kvec_M_plus_sign = S_kvec_M_BOTH[1]

    def body_fun_inner(i, val):
        logsum, sign, m = val
        sign_b = jnp.where(i%2==0, 1, -1)
        log_b = log_bk_t_θ_t(m+i, t, θ, m)
        logsum, sign = signed_logaddexp(logsum, log_b, sign1=sign, sign2=sign_b)
        return logsum, sign, m

    logsum_minus, sign_minus, _ = jax.lax.fori_loop(0, two_km_plus_1.astype(int) + 1, body_fun_inner, (-jnp.inf * jnp.ones_like(t), jnp.ones_like(t), M))
    log_S_kvec_M_minus, log_S_kvec_M_minus_sign = signed_logaddexp(log_S_kvec_M_minus, logsum_minus, sign1=log_S_kvec_M_minus_sign, sign2=sign_minus)

    log_b = log_bk_t_θ_t(M+two_km_plus_1, t, θ, M)
    logsum_plus, sign_plus = signed_logaddexp(log_b, logsum_minus, sign2=sign_minus)
    log_S_kvec_M_plus, log_S_kvec_M_plus_sign = signed_logaddexp(log_S_kvec_M_plus, logsum_plus, sign1=log_S_kvec_M_plus_sign, sign2=sign_plus)

    return (log_S_kvec_M_minus, log_S_kvec_M_minus_sign), (log_S_kvec_M_plus, log_S_kvec_M_plus_sign)

def S_kvec_M_both_logsumexp_addk(kvec, t, θ, M, S_kvec_M_BOTH):
    two_kvec_plus_1 = 2*kvec + 1
    
    log_S_kvec_M_minus, log_S_kvec_M_minus_sign = S_kvec_M_BOTH[0]
    log_S_kvec_M_plus, log_S_kvec_M_plus_sign = S_kvec_M_BOTH[1]

    # Compute logS_kvec_M_minus
    def body_fun_outer_1(m, val):
        logsum, sign = val
        two_km_plus_1 = two_kvec_plus_1[..., m]
        log_b_two_k = log_bk_t_θ_t(m+two_km_plus_1-1, t, θ, m)
        log_b_two_k_plus_1 = log_bk_t_θ_t(m+two_km_plus_1, t, θ, m)
        log_b, sign_b = signed_logaddexp(log_b_two_k, log_b_two_k_plus_1, sign2=-1)
        logsum, sign = signed_logaddexp(logsum, log_b, sign1=sign, sign2=sign_b)
        return logsum, sign

    log_S_kvec_M_minus, log_S_kvec_M_minus_sign = jax.lax.fori_loop(0, M.astype(int) + 1, body_fun_outer_1,  S_kvec_M_BOTH[0])

    # Compute logS_kvec_M_plus
    def body_fun_outer_2(m, val):
        logsum, sign = val
        two_km_plus_1 = two_kvec_plus_1[..., m]
        log_b_two_k = log_bk_t_θ_t(m+two_km_plus_1-1, t, θ, m)
        log_b_two_k_minus_1 = log_bk_t_θ_t(m+two_km_plus_1-2, t, θ, m)
        log_b, sign_b = signed_logaddexp(log_b_two_k, log_b_two_k_minus_1, sign2=-1)
        logsum, sign = signed_logaddexp(logsum, log_b, sign1=sign, sign2=sign_b)
        return logsum, sign

    log_S_kvec_M_plus, log_S_kvec_M_plus_sign = jax.lax.fori_loop(0, M.astype(int) + 1, body_fun_outer_2, S_kvec_M_BOTH[1])

    return (log_S_kvec_M_minus, log_S_kvec_M_minus_sign), (log_S_kvec_M_plus, log_S_kvec_M_plus_sign)


def Compute_A(rng, θ, t):
    t_thres = 0.1
    rng, step_rng = jax.random.split(rng)
    A_approx = Compute_A_approx(step_rng, θ, t)
    rng, step_rng = jax.random.split(rng)
    A_arb = Compute_A_good_start_arb(step_rng, θ, jnp.where(t<=t_thres, 1, t))
    # A_arb = jnp.ones(t.shape)
    return jnp.where(t<=t_thres, A_approx, A_arb)


def Wright_Fisher_K_dim_transition_with_t_small_approx(rng, xvec, t, theta_vec):
    import tensorflow_probability as tfp
    rng, rng0, rng1, rng2 = jax.random.split(rng, 4)
    # prng = jax.random.split(rng0, xvec.shape[0])
    stheta = jnp.sum(theta_vec, axis=-1)
    A = Compute_A(rng0, stheta, t)
    # print(A, xvec)
    L = tfp.substrates.jax.distributions.Multinomial(A, probs=xvec).sample(seed=rng1)
    Y = jax.random.dirichlet(rng2, L + theta_vec)
    return Y
