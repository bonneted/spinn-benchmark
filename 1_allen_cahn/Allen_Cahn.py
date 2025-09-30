"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle

Implementation of Allen-Cahn equation example in paper https://arxiv.org/abs/2111.02801.
"""
import deepxde as dde
import numpy as np
from scipy.io import loadmat
import jax
import jax.numpy as jnp

fourier_features = True  # True or False
net_type="FNN"  # "FNN" or "SPINN"
mlp_type = 'mlp'  # 'mlp' or 'modified-mlp', only for SPINN
# n_adaptive_sample = 10  # number of adaptive samples per iteration, 0 to disable


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
n_pde = 150**2
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
def fourier_features_transform(x, sigma=10, num_features=256):
    """Generate Fourier features for input x."""
    kernel = jax.random.normal(
        jax.random.PRNGKey(0), (x.shape[-1], num_features)
    ) * sigma
    y = jnp.concatenate(
        [jnp.cos(jnp.dot(x, kernel)), jnp.sin(jnp.dot(x, kernel))], axis=-1
    )
    return y

domain = np.array([[-1, 1], [0, 1]])
def adaptive_sampling_grid(domain, n, loss_fun, k=1, c=1, n_grid=200, 
                            random_state=None):
    """
    Choose n x-coordinates and n y-coordinates so that the n×n grid they
    generate (via Cartesian product) lies in the regions of highest loss.
    """
    rng = np.random.default_rng(random_state)

    # Build a random trial grid of shape (n_rand, n_rand)
    x_trial = rng.uniform(domain[0, 0], domain[0, 1], n_grid).reshape(-1, 1)  
    y_trial = rng.uniform(domain[1, 0], domain[1, 1], n_grid).reshape(-1, 1)

    # Evaluate the loss on every grid point
    loss_flat = loss_fun([x_trial, y_trial])
    loss = loss_flat.reshape(n_grid, n_grid)

    # Convert the loss into row / column scores
    weight = (loss ** k) / np.mean(loss ** k) + c    # emphasise large losses
    row_scores = weight.sum(axis=1)                  # shape (n_rand,)
    col_scores = weight.sum(axis=0)                  # shape (n_rand,)

    row_idx = np.argsort(-row_scores)[:n]
    col_idx = np.argsort(-col_scores)[:n]

    x_sample = np.sort(x_trial[row_idx])   # sort for nicer grids / plots
    y_sample = np.sort(y_trial[col_idx])

    return x_sample, y_sample

def PDE_loss(X):
    pde_loss_val = model.predict([X[0], X[1]], operator=pde)
    return np.abs(pde_loss_val)


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

n_hidden = 3
rank = 32
activations = "tanh" #["sin"]*(n_hidden) + ["sin"] + ["sin"]

if (net_type=="SPINN"):
    layers = [2] + [20] * n_hidden + [rank] + [1]
    net = dde.nn.SPINN(layers, activations, "Glorot normal", mlp_type)
else:
    net = dde.nn.FNN([2] + [20] * n_hidden + [1], activations, "Glorot normal")

net.apply_output_transform(output_transform)
if fourier_features:
    net.apply_feature_transform(fourier_features_transform)

model = dde.Model(data, net)
# data.replace_with_anchors(pde_anchors)
model.compile("adam", lr=1e-3)

learning_rates = [1e-3, 1e-4, 5e-5, 1e-5, 5e-6]
n_iters = 20000 #if net_type=="SPINN" else 30000
for lr in learning_rates:
    # if (n_adaptive_sample>0) and (net_type!="FNN"):
    #     x_sample, y_sample = adaptive_sampling_grid(domain,
    #                                         n=n_adaptive_sample,
    #                                         loss_fun=PDE_loss)
    #     pde_anchors = [np.concat((pde_anchors[0],x_sample)),
    #                     np.concat((pde_anchors[1],y_sample))]
    #     print(f"Added {n_adaptive_sample**2} adaptive samples, total {len(pde_anchors)} sets of anchors")
        # if len(pde_anchors) > args.max_adaptive_samples+1:
        #     del pde_anchors[1]
        # data.replace_with_anchors(pde_anchors)
    model.compile("adam", lr=lr)
    model.train(iterations=n_iters)

# model.compile("L-BFGS")
# losshistory, train_state = model.train()
# dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# X, y_true = gen_testdata()
X, y_true, xx, tt, u = gen_testdata()
u = u.T
y_pred = model.predict(X)
u_pred = y_pred.reshape(u.shape)
X_pde = X[0] if (net_type=="SPINN") else X
f = model.predict(X_pde, operator=pde)
pde_loss = f.reshape(u.shape)

# f = model.predict(X, operator=pde)

print(f"Mean pde residual: {np.nanmean(np.abs(pde_loss)):.3e}")
print(f"L2 relative error: {dde.metrics.l2_relative_error(u, u_pred):.3e}")
# np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,4, figsize=(24,4))

vmin = np.nanmin([u, u_pred])
vmax = np.nanmax([u, u_pred])

im = ax[0].pcolor(tt, xx, u, vmin=vmin, vmax=vmax)
fig.colorbar(im, ax=ax[0])
ax[0].set_title("Ground Truth")
ax[0].set_xlabel("t")
ax[0].set_ylabel("x")

im = ax[1].pcolor(tt, xx, u_pred, vmin=vmin, vmax=vmax)
fig.colorbar(im, ax=ax[1])
ax[1].set_title("Prediction")
ax[1].set_xlabel("t")
ax[1].set_ylabel("x")

im = ax[2].pcolor(tt,xx,np.absolute(u_pred - u), cmap="plasma")
fig.colorbar(im, ax=ax[2])
ax[2].set_title("Error")
ax[2].set_xlabel("t")
ax[2].set_ylabel("x")

im = ax[3].pcolor(tt,xx,np.absolute(pde_loss), cmap="plasma")
fig.colorbar(im, ax=ax[3])
ax[3].set_title("PDE residual")
ax[3].set_xlabel("t")
ax[3].set_ylabel("x")
plt.show()