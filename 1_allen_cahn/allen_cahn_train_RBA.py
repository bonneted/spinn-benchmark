import deepxde as dde
import numpy as np
from scipy.io import loadmat
import jax
import jax.numpy as jnp
import random
import time
import wandb


DEFAULT_CONFIG = {
    "fourier_features": True,
    "n_fourier_features": 128,
    "RBA": False,
    "sigma": 10,
    "net_type": "SPINN",   # or "PINN"
    "mlp_type": "mlp",
    "activations": "sin",
    "initialization": "Glorot normal",
    "n_hidden": 3,
    "rank": 64,
    "num_domain": 150**2,
    "n_iters": 30000,
    "seed": 0,
}

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def train_allen_cahn(config=None, wandb_project=None):
    # Always start from defaults
    cfg = DEFAULT_CONFIG.copy()

    # If user provided a config dict, override keys
    if config is not None:
        cfg.update(config)
    cfg = Config(**cfg)

    # if wandb logging, init with the merged config
    if wandb_project:
        import wandb
        wandb.init(project=wandb_project, config=cfg)
        cfg = wandb.config  # ensure cfg == wandb sweep config

    # -----------------------------
    # Set random seed
    # -----------------------------
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    if dde.backend.backend_name == "jax":
        key = jax.random.PRNGKey(cfg.seed)

    # -----------------------------
    # Extract hyperparameters
    # -----------------------------
    fourier_features = cfg.fourier_features
    n_fourier_features = cfg.n_fourier_features
    sigma = cfg.sigma
    net_type = cfg.net_type
    mlp_type = cfg.mlp_type
    activations = cfg.activations
    initialization = cfg.initialization
    n_hidden = cfg.n_hidden
    rank = cfg.rank
    n_pde = cfg.num_domain
    n_iters = cfg.n_iters
    d = 0.001

    @dde.utils.list_handler
    def transform_coords(x):
        x_mesh = [x_.ravel() for x_ in jnp.meshgrid(jnp.atleast_1d(x[0].squeeze()),
                                                     jnp.atleast_1d(x[1].squeeze()), indexing="ij")]
        return dde.backend.stack(x_mesh, axis=-1)

    if net_type == "SPINN":
        dde.config.set_default_autodiff("forward")
    else:
        dde.config.set_default_autodiff("reverse")

    cos = dde.backend.cos
    if dde.backend.backend_name == "jax":
        jax.config.update("jax_default_matmul_precision", "highest")

    # -----------------------------
    # Test data
    # -----------------------------
    def gen_testdata():
        data = loadmat("../dataset/Allen_Cahn.mat")
        t = data["t"]
        x = data["x"]
        u = data["u"]

        xx, tt = np.meshgrid(x, t, indexing="ij")
        X = np.vstack((np.ravel(xx), np.ravel(tt))).T
        y = u.flatten()[:, None]

        X_input = [[x.reshape(-1,1), t.reshape(-1,1)]] if (net_type=="SPINN") else X
        return X_input, y, xx, tt, u

    # -----------------------------
    # Residual-Based Attention
    # -----------------------------
    callbacks = []
    if cfg.RBA: #RBA initialization
        eta = 0.001
        # note that if eta = 0.01 the rba upper bound is 10 
        # so a multiplier of 10^2 is used for the initial condition
        gamma = 0.999
        rsum = 0

    # -----------------------------
    # PDE / SPINN helper functions
    # -----------------------------
    def pde_pinn(x, y, unknowns=None):

        if net_type!="PINN":
            x = transform_coords(x)
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        if dde.backend.backend_name == "jax":
            y, dy_t, dy_xx = y[0], dy_t[0], dy_xx[0]

        loss = dy_t - d * dy_xx - 5 * (y - y**3)
        if cfg.RBA and unknowns is not None: #RBA weight update
            rsum = unknowns[0]
            # r_norm = eta * jnp.abs(loss)/jnp.max(jnp.abs(loss))
            # rsum = rsum * gamma + r_norm 
            loss = rsum * loss
        return loss

    def hvp_fwdfwd(f, x, tangents, return_primals=False):
        g = lambda primals: jax.jvp(f, (primals,), tangents)[1]
        primals_out, tangents_out = jax.jvp(g, x, tangents)
        return (primals_out, tangents_out) if return_primals else tangents_out

    def pde_spinn(X, y, unknowns=None):
        x, t = X[0].reshape(-1, 1), X[1].reshape(-1, 1)
        v_x = jnp.ones_like(x)
        v_t = jnp.ones_like(t)

        u = y[0]
        dy_t = jax.jvp(lambda t: y[1]((x,t)), (t,), (v_t,))[1]
        dy_xx = hvp_fwdfwd(lambda x: y[1]((x,t)), (x,), (v_x,))
        loss = dy_t - d * dy_xx - 5 * (u - u**3)
        if cfg.RBA and unknowns is not None: #RBA weight update
            rsum = unknowns[0]
            # r_norm = eta * jnp.abs(loss)/jnp.max(jnp.abs(loss))
            # rsum = rsum * gamma + r_norm 
            loss = rsum * loss
        return loss

    pde_fn = pde_pinn if net_type=="PINN" else pde_spinn
    pde_no_rba = lambda x, y: pde_fn(x, y, unknowns=None)

    if cfg.RBA:
        residual_op = lambda x, y: pde(x, y, unknowns=None)
        RBA_callback = dde.callbacks.ResidualBasedAttention(residual_op, eta=eta, gamma=gamma)
        callbacks.append(RBA_callback)
        pde = pde_fn
    else :
        pde = pde_no_rba
    # -----------------------------
    # Fourier feature transform
    # -----------------------------
    def list_handler(func):
        def wrapper(x, *args, **kwargs):
            if isinstance(x, (list, tuple)):
                return [func(xi.reshape(-1,1), *args, **kwargs) for xi in x]
            return func(x, *args, **kwargs)
        return wrapper

    @list_handler
    def fourier_features_transform(x, sigma=sigma, num_features=n_fourier_features):
        kernel = jax.random.normal(jax.random.PRNGKey(0), (x.shape[-1], num_features)) * sigma
        y = jnp.concatenate([jnp.cos(jnp.dot(x, kernel)), jnp.sin(jnp.dot(x, kernel))], axis=-1)
        return y

    # -----------------------------
    # Output transform
    # -----------------------------
    def output_transform(x, y):
        if net_type!="PINN" and isinstance(x, (list, tuple)):
            x = transform_coords(x)
        out = x[:, 0:1]**2 * cos(np.pi * x[:, 0:1]) + x[:, 1:2] * (1 - x[:, 0:1]**2) * y.reshape(-1,1)
        return out

    # -----------------------------
    # Geometry / Data
    # -----------------------------
    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    if net_type != "PINN":
        x_all = np.linspace(-1, 1, int(np.sqrt(n_pde))).reshape(-1, 1)
        t_all = np.linspace(0, 1, int(np.sqrt(n_pde))).reshape(-1, 1)
        geomtime = dde.geometry.ListPointCloud([x_all, t_all])

    if net_type=="PINN":
        data = dde.data.TimePDE(geomtime, pde, [], num_domain=n_pde, num_boundary=0, num_initial=0)
    else:
        data = dde.data.PDE(geomtime, pde, [], num_domain=n_pde, num_boundary=0, is_SPINN=True)

    # -----------------------------
    # Neural network
    # -----------------------------
    if net_type=="SPINN":
        layers = [2] + [20] * n_hidden + [rank] + [1]
        net = dde.nn.SPINN(layers, activations, initialization, mlp_type)
    else:
        net = dde.nn.FNN([2] + [20] * n_hidden + [1], activations, initialization)

    net.apply_output_transform(output_transform)
    if fourier_features:
        net.apply_feature_transform(fourier_features_transform)

    model = dde.Model(data, net)

    # -----------------------------
    # Training
    # -----------------------------
    learning_rates = [1e-3, 1e-4, 5e-5, 1e-5, 5e-6]
    start_time = time.time()
    for lr in learning_rates:
        model.compile("adam", lr=lr)
        losshistory, train_state = model.train(iterations=n_iters, callbacks=callbacks)
    elapsed = time.time() - start_time
    its_per_sec = n_iters * 5 / elapsed

    # -----------------------------
    # Evaluation
    # -----------------------------
    X, y_true, xx, tt, u = gen_testdata()
    u = u.T
    y_pred = model.predict(X)
    u_pred = y_pred.reshape(u.shape)
    X_pde = X[0] if net_type=="SPINN" else X
    f = model.predict(X_pde, operator=pde_no_rba)
    pde_loss = f.reshape(u.shape)

    mean_pde_residual = float(np.nanmean(np.abs(pde_loss)))
    l2_error = float(dde.metrics.l2_relative_error(u, u_pred))

    print(f"Mean PDE residual: {mean_pde_residual:.3e}")
    print(f"L2 relative error: {l2_error:.3e}")
    print(f"Elapsed training time: {elapsed:.2f} s, {its_per_sec:.2f} it/s")

    # -----------------------------
    # Log to wandb if enabled
    # -----------------------------
    if wandb_project is not None:
        wandb.log({
            "mean_pde_residual": mean_pde_residual,
            "l2_relative_error": l2_error,
            "final_loss": float(train_state.loss_train[0]),
            "elapsed_time_s": elapsed,
            "iterations_per_sec": its_per_sec
        })
        wandb.finish()

    return {
        "model": model,
        "u_pred": u_pred,
        "u_true": u,
        "pde_loss": pde_loss,
        "l2_error": l2_error,
        "mean_pde_residual": mean_pde_residual,
        "elapsed_time": elapsed,
        "iterations_per_sec": its_per_sec
    }


if __name__ == "__main__":
    results = train_allen_cahn()
