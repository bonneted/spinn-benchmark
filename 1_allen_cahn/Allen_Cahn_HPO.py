import deepxde as dde
import numpy as np
from scipy.io import loadmat
import jax
import jax.numpy as jnp
import wandb
import random
import os
import time

# -----------------------------
# Hyperparameters (tracked by wandb)
# -----------------------------
hyperparams = dict(
    fourier_features=True,          # True or False
    n_fourier_features=256,        # number of fourier features
    sigma=10,                       # stddev of fourier feature kernel
    net_type="FNN",                 # "FNN" or "SPINN"
    mlp_type="mlp",                 # "mlp" or "modified-mlp"
    activations="tanh",             # string
    n_hidden=3,
    rank=32,
    num_domain=150**2,
    n_iters=20000,
    seed=42                         # for reproducibility / sweep override
)

wandb.init(project="Allen-Cahn-SPINN", config=hyperparams)
config = wandb.config

# -----------------------------
# Set random seed
# -----------------------------
np.random.seed(config.seed)
random.seed(config.seed)
if dde.backend.backend_name == "jax":
    key = jax.random.PRNGKey(config.seed)

# -----------------------------
# Rest of your code, modified to use config
# -----------------------------

fourier_features = config.fourier_features
n_fourier_features = config.n_fourier_features
sigma = config.sigma

net_type = config.net_type
mlp_type = config.mlp_type
activations = config.activations
n_hidden = config.n_hidden
rank = config.rank
n_pde = config.num_domain

@dde.utils.list_handler
def transform_coords(x):
    x_mesh = [x_.ravel() for x_ in jnp.meshgrid(jnp.atleast_1d(x[0].squeeze()), jnp.atleast_1d(x[1].squeeze()), indexing="ij")]
    return dde.backend.stack(x_mesh, axis=-1)

if net_type=="SPINN":
    dde.config.set_default_autodiff("forward")
else:
    dde.config.set_default_autodiff("reverse")

cos = dde.backend.cos

if dde.backend.backend_name == "jax":
    import jax
    jax.config.update("jax_default_matmul_precision", "highest")

def gen_testdata():
    data = loadmat("./dataset/Allen_Cahn.mat")

    t = data["t"]
    x = data["x"]
    u = data["u"]

    dt = dx = 0.01
    xx, tt = np.meshgrid(x, t, indexing="ij")
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = u.flatten()[:, None]

    X_input = [[x.reshape(-1,1), t.reshape(-1,1)]] if (net_type=="SPINN") else X
    return X_input, y, xx, tt, u

geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)
d = 0.001
if (net_type!="FNN"):

    x_all = np.linspace(-1, 1, int(np.sqrt(n_pde))).reshape(-1, 1)
    t_all = np.linspace(0, 1, int(np.sqrt(n_pde))).reshape(-1, 1)
    pde_anchors = [x_all, t_all]
    geomtime = dde.geometry.ListPointCloud(pde_anchors)

def pde(x, y):
    if (net_type!="FNN"):
        x = transform_coords(x)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    if dde.backend.backend_name == "jax":
        y, dy_t, dy_xx = y[0], dy_t[0], dy_xx[0] # second element is pure function
    return dy_t - d * dy_xx - 5 * (y - y**3)

def hvp_fwdfwd(f, x, tangents, return_primals=False):
    g = lambda primals: jax.jvp(f, (primals,), tangents)[1]
    primals_out, tangents_out = jax.jvp(g, x, tangents)
    if return_primals:
        return primals_out, tangents_out
    else:
        return tangents_out
    
def pde_spinn(X, y):
    x, t = X[0].reshape(-1, 1), X[1].reshape(-1, 1)
    v_x = jnp.ones_like(x)
    v_t = jnp.ones_like(t)

    u = y[0] 
    dy_t = jax.jvp(lambda t: y[1]((x,t)), (t,), (v_t,))[1] # ∂u/∂t
    dy_xx = hvp_fwdfwd(lambda x: y[1]((x,t)), (x,), (v_x,))  # ∂²u/∂x²
    return dy_t - d * dy_xx - 5 * (u - u**3)

pde = pde if (net_type=="FNN") else pde_spinn

# Fourier feature transform

def list_handler(func):
    def wrapper(x, *args, **kwargs):
        if isinstance(x, (list, tuple)):
            return [func(xi.reshape(-1, 1), *args, **kwargs) for xi in x]
        return func(x, *args, **kwargs)
    return wrapper

@list_handler
def fourier_features_transform(x, sigma=sigma, num_features=n_fourier_features):
    """Generate Fourier features for input x."""
    kernel = jax.random.normal(
        jax.random.PRNGKey(0), (x.shape[-1], num_features)
    ) * sigma
    y = jnp.concatenate(
        [jnp.cos(jnp.dot(x, kernel)), jnp.sin(jnp.dot(x, kernel))], axis=-1
    )
    return y


# Hard restraints on initial + boundary conditions
def output_transform(x, y):
    if (net_type!="FNN") and isinstance(x, (list,tuple)):
        x = transform_coords(x)
    out = x[:, 0:1]**2 * cos(np.pi * x[:, 0:1]) + x[:, 1:2] * (1 - x[:, 0:1]**2) * y.reshape(-1,1)
    return out

if (net_type=="FNN"):
    data = dde.data.TimePDE(geomtime, pde, [], num_domain=n_pde, num_boundary=0, num_initial=0)
else:
    data = dde.data.PDE(geomtime, pde, [], num_domain=n_pde, num_boundary=0, is_SPINN=True)

if (net_type=="SPINN"):
    layers = [2] + [20] * n_hidden + [rank] + [1]
    net = dde.nn.SPINN(layers, activations, "Glorot normal", mlp_type)
else:
    net = dde.nn.FNN([2] + [20] * n_hidden + [1], activations, "Glorot normal")

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
    losshistory, train_state = model.train(iterations=config.n_iters)
elapsed = time.time() - start_time
its_per_sec = config.n_iters*5 / elapsed

# -----------------------------
# Evaluation
# -----------------------------
X, y_true, xx, tt, u = gen_testdata()
u = u.T
y_pred = model.predict(X)
u_pred = y_pred.reshape(u.shape)
X_pde = X[0] if (net_type=="SPINN") else X
f = model.predict(X_pde, operator=pde)
pde_loss = f.reshape(u.shape)

mean_pde_residual = float(np.nanmean(np.abs(pde_loss)))
l2_error = float(dde.metrics.l2_relative_error(u, u_pred))

print(f"Mean pde residual: {mean_pde_residual:.3e}")
print(f"L2 relative error: {l2_error:.3e}")
print(f"Elapsed training time: {elapsed:.2f} s, {its_per_sec:.2f} it/s")

# -----------------------------
# Log to wandb
# -----------------------------
wandb.log({
    "mean_pde_residual": mean_pde_residual,
    "l2_relative_error": l2_error,
    "final_loss": float(train_state.loss_train[0]),
    "elapsed_time_s": elapsed,
    "iterations_per_sec": its_per_sec,
})

