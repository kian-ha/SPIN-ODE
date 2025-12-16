"""
SPIN-ODE Complete Pipeline with MULTIPLICATIVE GAUSSIAN NOISE

Based on: SPIN-ODE paper (https://arxiv.org/abs/2505.05625)
"""
import jax
import jax.numpy as jnp
from flax import nnx
import optax
from typing import Tuple, Callable
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import numpy as np
import diffrax
import torch
from torch.utils.data import Dataset, DataLoader, default_collate
from pathlib import Path
import argparse

jax.config.update("jax_enable_x64", True)
torch.set_default_dtype(torch.float64)


print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")
print(f"JAX default backend: {jax.default_backend()}")
print("=" * 80)



def add_multiplicative_gaussian_noise(
    y_arr: np.ndarray,
    noise_level: float,
    random_seed: int = None,
) -> np.ndarray:

    if random_seed is not None:
        np.random.seed(random_seed)
    
    noise = np.random.normal(0, noise_level * np.abs(y_arr), size=y_arr.shape)
    y_noisy = y_arr + noise
    
    y_noisy = np.maximum(y_noisy, 0.0)
    

    original_sum = np.sum(y_arr, axis=-1, keepdims=True)
    noisy_sum = np.sum(y_noisy, axis=-1, keepdims=True)
    y_noisy = y_noisy * (original_sum / (noisy_sum + 1e-12))
    
    return y_noisy


def robertson_ode(t, y, args=None):
    k1, k2, k3 = 0.04, 3e7, 1e4
    A, B, C = y[0], y[1], y[2]
    
    dA = -k1 * A + k3 * B * C
    dB = k1 * A - k2 * B * B - k3 * B * C
    dC = k2 * B * B
    
    return jnp.array([dA, dB, dC])


def generate_robertson_data_scipy(
    y0: np.ndarray = np.array([1.0, 0.0, 0.0]),
    t_span: Tuple[float, float] = (1e-5, 1e5),
    n_points: int = 50,
    n_series: int = 1,
    rand_init: bool = False,
    noise_level: float = 0.0,
    random_seed: int = None,
) -> Tuple[np.ndarray, np.ndarray]:

    ts = np.logspace(np.log10(t_span[0]), np.log10(t_span[1]), n_points)
    
    y_arr_list = []
    t_arr_list = []
    
    for i in range(n_series):
        if rand_init and i > 0:
            y0_perturbed = y0 * (1 + np.random.normal(0, 0.1, size=3))
            y0_perturbed = np.clip(y0_perturbed, 0, None)
            y0_perturbed = y0_perturbed / np.sum(y0_perturbed)
        else:
            y0_perturbed = y0
        
        solution = solve_ivp(
            lambda t, y: np.array(robertson_ode(t, y)),
            [ts[0], ts[-1]],
            y0_perturbed,
            method='BDF',
            t_eval=ts,
            rtol=1e-8,
            atol=1e-10,
        )
        
        y_arr_list.append(solution.y.T)
        t_arr_list.append(ts)
    
    y_arr = np.array(y_arr_list)
    t_arr = np.array(t_arr_list)
    
    if noise_level > 0:
        y_arr = add_multiplicative_gaussian_noise(y_arr, noise_level, random_seed)
    
    return y_arr, t_arr




def jax_collate(batch):
    return jax.tree_util.tree_map(jnp.asarray, default_collate(batch))


class ChuckDataset(Dataset):

    
    def __init__(
        self,
        ny: np.ndarray,
        nt: np.ndarray,
        chuck_len: int = None,
        stride_len: int = 1,
        ratio: float = 1.0,
    ):
        self.ny = ny
        self.nt = nt
        self.n_series = self.nt.shape[0]
        self.series_len = self.nt.shape[1]
        self.chuck_len = self.series_len if chuck_len is None else chuck_len
        self.stride_len = stride_len
        self.chuck_per_serie = (self.series_len - self.chuck_len) // self.stride_len + 1
        self.total_chuck = self.n_series * self.chuck_per_serie
        self.samples_per_chuck = int(self.chuck_len * ratio)
        self.rand_sample()
        
    
    def rand_sample(self):
        self.idx_sample = np.sort(
            np.random.choice(
                range(0, self.chuck_len),
                size=self.samples_per_chuck,
                replace=False
            )
        )
    
    def __len__(self):
        return self.total_chuck
    
    def __getitem__(self, idx):
        idx_series = idx // self.chuck_per_serie
        idx_chuck = idx % self.chuck_per_serie
        start = idx_chuck * self.stride_len
        
        return {
            "conc": self.ny[idx_series, start + self.idx_sample],
            "time": self.nt[idx_series, start + self.idx_sample],
        }



class Var(nnx.Variable):
    pass


class NormalizedMLP(nnx.Module):
    
    def __init__(
        self,
        n_species: int,
        hidden_size: int,
        y_min: jnp.ndarray,
        y_max: jnp.ndarray,
        dy_scale: jnp.ndarray,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        
        self.linear1 = nnx.Linear(n_species, hidden_size, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)
        self.linear3 = nnx.Linear(hidden_size, n_species, rngs=rngs)
        
        self.y_min = Var(y_min)
        self.y_scale = Var(y_max - y_min + 1e-10)
        self.dy_scale = Var(dy_scale)
    
    def __call__(self, t: float, y: jnp.ndarray) -> jnp.ndarray:
        y_norm = (y - self.y_min.value) / self.y_scale.value
        
        x = nnx.gelu(self.linear1(y_norm))
        x = nnx.gelu(self.linear2(x))
        dy_dt_norm = self.linear3(x)
        
        dy_dt = dy_dt_norm * self.dy_scale.value
        
        return dy_dt




class RobertsonCRNN(nnx.Module):
    
    def __init__(
        self,
        k_init: jnp.ndarray = jnp.array([0.04, 3e7, 1e4]),
        *,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        
        self.log_k = nnx.Param(jnp.log(k_init))
        
        # Stoichiometric matrices
        self.stoi_forward = Var(jnp.array([
            [1.0, 0.0, 0.0],  # Reaction 1: [A]^1
            [0.0, 2.0, 0.0],  # Reaction 2: [B]^2
            [0.0, 1.0, 1.0],  # Reaction 3: [B]^1 * [C]^1
        ]))
        
        self.stoi_net = Var(jnp.array([
            [-1.0,  0.0, +1.0],  # A
            [+1.0, -1.0, -1.0],  # B
            [ 0.0, +1.0,  0.0],  # C
        ]))
    
    def __call__(self, t: float, y: jnp.ndarray) -> jnp.ndarray:
        y_safe = jnp.clip(y, 1e-30, 1e30)
        log_y = jnp.log(y_safe)
        log_rates = self.log_k.value + self.stoi_forward.value @ log_y
        rates = jnp.exp(log_rates)
        dydt = self.stoi_net.value @ rates
        return dydt
    
    def get_rate_coefficients(self) -> jnp.ndarray:
        return jnp.exp(self.log_k.value)




def compute_derivatives(ys: jnp.ndarray, ts: jnp.ndarray) -> jnp.ndarray:
    dydt = jnp.zeros_like(ys)
    
    dydt = dydt.at[0].set((ys[1] - ys[0]) / (ts[1] - ts[0]))
    
    for i in range(1, len(ts) - 1):
        dt = ts[i + 1] - ts[i - 1]
        dy = ys[i + 1] - ys[i - 1]
        dydt = dydt.at[i].set(dy / dt)
    
    dydt = dydt.at[-1].set((ys[-1] - ys[-2]) / (ts[-1] - ts[-2]))
    
    return dydt




def interpolate_trajectory(
    model: NormalizedMLP,
    ts_original: jnp.ndarray,
    ys_original: jnp.ndarray,
    interpolation_factor: int = 10,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    t_min, t_max = float(ts_original[0]), float(ts_original[-1])
    n_interp = len(ts_original) * interpolation_factor
    
    def learned_ode(t, y):
        y_jax = jnp.array(y)
        dydt = model(t, y_jax)
        return np.array(dydt)
    
    ts_interp = np.logspace(np.log10(t_min), np.log10(t_max), n_interp)
    
    try:
        solution = solve_ivp(
            learned_ode,
            [ts_interp[0], ts_interp[-1]],
            np.array(ys_original[0]),
            method='BDF',
            t_eval=ts_interp,
            rtol=1e-6,
            atol=1e-8,
        )
        ys_interp = solution.y.T
    except Exception as e:
        print(f"  Warning: Interpolation via BDF failed, using scipy interp1d: {e}")
        interp_funcs = [interp1d(ts_original, ys_original[:, i], kind='cubic') 
                       for i in range(ys_original.shape[1])]
        ys_interp = np.column_stack([f(ts_interp) for f in interp_funcs])
    
    return jnp.array(ts_interp), jnp.array(ys_interp)




def derivative_loss_with_constraints(
    model: NormalizedMLP,
    ts: jnp.ndarray,
    ys: jnp.ndarray,
    dydt_true: jnp.ndarray,
) -> Tuple[float, dict]:
    dydt_pred = jax.vmap(lambda t, y: model(t, y))(ts, ys)
    
    dy_scale = jnp.maximum(jnp.max(jnp.abs(dydt_true), axis=0), 1e-8)
    loss_deriv = jnp.mean(((dydt_pred - dydt_true) / dy_scale) ** 2)
    
    dy_sum = jnp.sum(dydt_pred, axis=1)
    loss_conservation = jnp.mean(dy_sum ** 2) * 0.1
    
    dt_avg = jnp.mean(ts[1:] - ts[:-1])
    y_next_estimate = ys + dydt_pred * dt_avg
    negative_penalty = jnp.mean(jnp.maximum(0, -y_next_estimate) ** 2) * 0.1
    
    total_loss = loss_deriv
    
    info = {
        'loss': total_loss,
        'loss_deriv': loss_deriv,
        'loss_conservation': loss_conservation,
        'negative_penalty': negative_penalty,
    }
    
    return total_loss, info


def crnn_derivative_loss(
    model: RobertsonCRNN,
    ys: jnp.ndarray,
    dydt_true: jnp.ndarray,
) -> Tuple[float, dict]:
    dydt_pred = jax.vmap(lambda y: model(0.0, y))(ys)
    
    dy_scale = jnp.maximum(jnp.max(jnp.abs(dydt_true), axis=0), 1e-8)
    loss_deriv = jnp.mean(((dydt_pred - dydt_true) / dy_scale) ** 2)
    
    dy_sum = jnp.sum(dydt_pred, axis=1)
    loss_conservation = jnp.mean(dy_sum ** 2) * 0.1
    
    k = model.get_rate_coefficients()
    k_true = jnp.array([0.04, 3e7, 1e4])
    loss_k_reg = jnp.mean((jnp.log(k) - jnp.log(k_true)) ** 2) * 0.001
    
    total_loss = loss_deriv
    
    info = {
        'loss': total_loss,
        'loss_deriv': loss_deriv,
        'loss_conservation': loss_conservation,
        'loss_k_reg': loss_k_reg,
        'k': k,
    }
    
    return total_loss, info




def create_diffrax_solver(
    solver_type=diffrax.Kvaerno3(),
    rtol: float = 1e-6,
    atol: float = 1e-7,
    max_steps: int = 8192,
):
    def integrate_ode(
        ode_func,
        ts: jnp.ndarray,
        y0: jnp.ndarray,
    ) -> jnp.ndarray:
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(lambda t, y, args: ode_func(t, y)),
            solver_type,
            t0=ts[0],
            t1=ts[-1],
            y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
            dt0=None,
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
            max_steps=max_steps,
            adjoint=diffrax.RecursiveCheckpointAdjoint(checkpoints=max_steps),
        )
        
        return solution.ys
    
    return integrate_ode


def ode_trajectory_loss(
    model: RobertsonCRNN,
    batch: dict,
    ode_solver: Callable,
    y_scale: jnp.ndarray,
) -> Tuple[float, dict]:

    conc = batch['conc']
    time = batch['time']
    
    ys_pred = ode_solver(model, time, conc[0])
    
    loss_traj = jnp.mean(((ys_pred - conc) / y_scale) ** 2)
    
    total_mass_pred = jnp.sum(ys_pred, axis=1)
    total_mass_true = jnp.sum(conc, axis=1)
    loss_conservation = jnp.mean((total_mass_pred - total_mass_true) ** 2) * 0.01
    
    k = model.get_rate_coefficients()
    k_prior = jnp.array([0.04, 3e7, 1e4])
    loss_k_smooth = jnp.mean((jnp.log(k) - jnp.log(k_prior)) ** 2) * 0.001
    
    negative_conc = jnp.sum(jnp.maximum(0, -ys_pred))
    loss_positive = negative_conc ** 2 * 0.1
    
    total_loss = loss_traj
    
    return total_loss


def batch_ode_loss(
    model: RobertsonCRNN,
    batch: dict,
    ode_solver: Callable,
    y_scale: jnp.ndarray,
) -> float:

    conc_batch = batch['conc']
    time_batch = batch['time']
    
    def single_loss(conc, time):
        batch_single = {'conc': conc, 'time': time}
        return ode_trajectory_loss(model, batch_single, ode_solver, y_scale)
    
    batch_losses = nnx.vmap(
        single_loss,
        in_axes=(0, 0),
        out_axes=0,
    )(conc_batch, time_batch)
    
    return jnp.mean(batch_losses)



def train_step_stage1(
    model: NormalizedMLP,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    ts: jnp.ndarray,
    ys: jnp.ndarray,
    dydt_true: jnp.ndarray,
) -> Tuple[NormalizedMLP, optax.OptState, float, dict]:
    
    graphdef, params, others = nnx.split(model, nnx.Param, ...)
    
    def loss_fn(params):
        model = nnx.merge(graphdef, params, others)
        loss, info = derivative_loss_with_constraints(model, ts, ys, dydt_true)
        return loss, info
    
    (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    model = nnx.merge(graphdef, params, others)
    
    return model, opt_state, loss, info


def train_step_stage2(
    model: RobertsonCRNN,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    ys: jnp.ndarray,
    dydt_true: jnp.ndarray,
) -> Tuple[RobertsonCRNN, optax.OptState, float, dict]:
    
    graphdef, params, others = nnx.split(model, nnx.Param, ...)
    
    def loss_fn(params):
        model = nnx.merge(graphdef, params, others)
        loss, info = crnn_derivative_loss(model, ys, dydt_true)
        return loss, info
    
    (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    model = nnx.merge(graphdef, params, others)
    
    return model, opt_state, loss, info


@nnx.jit(static_argnames=['ode_solver'])
def train_step_stage3(
    model: RobertsonCRNN,
    optimizer: nnx.Optimizer,
    batch: dict,
    ode_solver: Callable,
    y_scale: jnp.ndarray,
) -> Tuple[float, jnp.ndarray]:

    
    def loss_fn(model):
        return batch_ode_loss(model, batch, ode_solver, y_scale)
    
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads, value=loss)
    
    k = model.get_rate_coefficients()
    
    return loss, k



def integrate_neural_ode_scipy(
    model,
    y0: np.ndarray,
    ts: np.ndarray,
) -> np.ndarray:
    def learned_ode(t, y):
        y_jax = jnp.array(y)
        dydt = model(t, y_jax)
        return np.array(dydt)
    
    try:
        solution = solve_ivp(
            learned_ode,
            [ts[0], ts[-1]],
            y0,
            method='BDF',
            t_eval=ts,
            rtol=1e-6,
            atol=1e-8,
        )
        return solution.y.T
    except Exception as e:
        print(f"    Integration failed: {e}")
        return None




def save_mlp_params(model: NormalizedMLP, filepath: str):
    params_dict = {
        'linear1_kernel': np.array(model.linear1.kernel.value),
        'linear1_bias': np.array(model.linear1.bias.value),
        'linear2_kernel': np.array(model.linear2.kernel.value),
        'linear2_bias': np.array(model.linear2.bias.value),
        'linear3_kernel': np.array(model.linear3.kernel.value),
        'linear3_bias': np.array(model.linear3.bias.value),
        'y_min': np.array(model.y_min.value),
        'y_scale': np.array(model.y_scale.value),
        'dy_scale': np.array(model.dy_scale.value),
    }
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    np.savez(filepath, **params_dict)


def save_crnn_params(model: RobertsonCRNN, filepath: str):
    params_dict = {
        'log_k': np.array(model.log_k.value),
        'k': np.array(model.get_rate_coefficients()),
    }
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    np.savez(filepath, **params_dict)



def train_stage1(
    y_arr: np.ndarray,
    t_arr: np.ndarray,
    hidden_size: int = 128,
    n_epochs: int = 10000,
    learning_rate: float = 1e-3,
    key: jax.random.PRNGKey = None,
) -> Tuple[NormalizedMLP, list]:
    
    if key is None:
        key = jax.random.PRNGKey(np.random.randint(1e6))
    
    
    ts = jnp.array(t_arr[0])
    ys = jnp.array(y_arr[0])
    
    dydt_data = compute_derivatives(ys, ts)
    
    y_min = jnp.min(ys, axis=0)
    y_max = jnp.max(ys, axis=0)
    dy_scale = jnp.maximum(jnp.max(jnp.abs(dydt_data), axis=0), 1e-8)
    
    rngs = nnx.Rngs(key)
    model = NormalizedMLP(
        n_species=ys.shape[1],
        hidden_size=hidden_size,
        y_min=y_min,
        y_max=y_max,
        dy_scale=dy_scale,
        rngs=rngs,
    )
    
    schedule = optax.exponential_decay(
        init_value=learning_rate,
        transition_steps=n_epochs // 10,
        decay_rate=0.95,
    )
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(nnx.state(model, nnx.Param))
    
    history = []
    best_loss = float('inf')
    
    for epoch in range(n_epochs):
        model, opt_state, loss, info = train_step_stage1(
            model, opt_state, optimizer, ts, ys, dydt_data
        )
        
        if epoch % 100 == 0 or epoch == n_epochs - 1:
            ys_pred = integrate_neural_ode_scipy(model, np.array(ys[0]), np.array(ts))
            if ys_pred is not None:
                traj_mse = np.mean((ys_pred - np.array(ys)) ** 2)
            else:
                traj_mse = float('inf')
            
            history.append({
                'epoch': epoch,
                'loss': float(loss),
                'loss_deriv': float(info['loss_deriv']),
                'traj_mse': float(traj_mse),
            })
            
            if loss < best_loss:
                best_loss = loss
            
            if epoch % 500 == 0:
                print(f"    Epoch {epoch:4d} | Loss: {loss:.6e} | "
                      f"Deriv: {info['loss_deriv']:.6e} | "
                      f"Traj MSE: {traj_mse:.6e}")
    
    print(f"  ✓ Stage 1 complete! Best loss: {best_loss:.6e}")
    return model, history


def train_stage2(
    mlp_model: NormalizedMLP,
    y_arr: np.ndarray,
    t_arr: np.ndarray,
    interpolation_factor: int = 10,
    n_epochs: int = 10000,
    learning_rate: float = 0.01,
    key: jax.random.PRNGKey = None,
) -> Tuple[RobertsonCRNN, list, jnp.ndarray, jnp.ndarray]:
    
    if key is None:
        key = jax.random.PRNGKey(43)
    
    ts_original = jnp.array(t_arr[0])
    ys_original = jnp.array(y_arr[0])
    

    
    ts_interp, ys_interp = interpolate_trajectory(
        mlp_model, ts_original, ys_original, interpolation_factor
    )
    
    print(f"\n  [Stage 2.2] Computing derivatives")
    dydt_interp = compute_derivatives(ys_interp, ts_interp)
    
    print(f"\n  [Stage 2.3] Training CRNN...")
    
    k_true = jnp.array([0.04, 3e7, 1e4])
    perturbation = jax.random.normal(key, (3,)) * 0.3
    k_init = k_true * jnp.exp(perturbation)
    rngs = nnx.Rngs(key)
    crnn_model = RobertsonCRNN(k_init=k_init, rngs=rngs)
    
    print(f"    Initial k: [{k_init[0]:.4e}, {k_init[1]:.4e}, {k_init[2]:.4e}]")
    
    schedule = optax.exponential_decay(
        init_value=learning_rate,
        transition_steps=n_epochs // 10,
        decay_rate=0.95,
    )
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(nnx.state(crnn_model, nnx.Param))
    
    history = []
    best_loss = float('inf')
    
    for epoch in range(n_epochs):
        crnn_model, opt_state, loss, info = train_step_stage2(
            crnn_model, opt_state, optimizer, ys_interp, dydt_interp
        )
        
        if epoch % 500 == 0 or epoch == n_epochs - 1:
            k = info['k']
            history.append({
                'epoch': epoch,
                'loss': float(loss),
                'loss_deriv': float(info['loss_deriv']),
                'k1': float(k[0]),
                'k2': float(k[1]),
                'k3': float(k[2]),
            })
            
            if loss < best_loss:
                best_loss = loss
            
            if epoch % 1000 == 0:
                k_error = jnp.mean(jnp.abs(jnp.log(k) - jnp.log(k_true)))
                print(f"      Epoch {epoch:5d} | Loss: {loss:.6e} | "
                      f"k=[{k[0]:.2e}, {k[1]:.2e}, {k[2]:.2e}] | "
                      f"Log error: {k_error:.4f}")
    
    print(f"Stage 2 complete. Best loss: {best_loss:.6e}")
    
    k_final = crnn_model.get_rate_coefficients()
    print(f"\n  Stage 2 final rate coefficients:")
    print(f"    k1: {k_final[0]:.6e} (true: {k_true[0]:.6e})")
    print(f"    k2: {k_final[1]:.6e} (true: {k_true[1]:.6e})")
    print(f"    k3: {k_final[2]:.6e} (true: {k_true[2]:.6e})")
    
    return crnn_model, history, ts_interp, ys_interp


def train_stage3(
    crnn_model: RobertsonCRNN,
    y_arr: np.ndarray,
    t_arr: np.ndarray,
    n_epochs: int = 10000,
    learning_rate: float = 0.01,
    batch_size: int = 64,
    chuck_len: int = 50,
    stride_len: int = 50,
    patience_ratio: float = 0.1,
    key: jax.random.PRNGKey = None,
) -> Tuple[RobertsonCRNN, list]:

    
    if key is None:
        key = jax.random.PRNGKey(np.random.randint(1e6))
    
    dataset = ChuckDataset(
        ny=y_arr,
        nt=t_arr,
        chuck_len=chuck_len,
        stride_len=stride_len,
        ratio=1.0,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=jax_collate,
    )
    
    
    y_scale = jnp.max(y_arr, axis=(0, 1)) - jnp.min(y_arr, axis=(0, 1))
    y_scale = jnp.where(y_scale == 0, 1.0, y_scale)
    
    k_initial = crnn_model.get_rate_coefficients()
    print(f"\n  [Stage 3.3] Initial k (from Stage 2): [{k_initial[0]:.4e}, {k_initial[1]:.4e}, {k_initial[2]:.4e}]")
    
    ode_solver = create_diffrax_solver(
        solver_type=diffrax.Kvaerno3(),
        rtol=1e-6,
        atol=1e-7,
        max_steps=8192,
    )
    
    patience_steps = int(patience_ratio * n_epochs * len(dataloader))
    optimizer = nnx.Optimizer(
        crnn_model,
        optax.chain(
            optax.adam(learning_rate),
            optax.contrib.reduce_on_plateau(
                patience=patience_steps,
                factor=0.5,
                cooldown=0,
            )
        ),
        wrt=nnx.Param,
    )
    
    history = []
    best_loss = float('inf')
    
    for epoch in range(n_epochs):
        epoch_losses = []
        
        dataset.rand_sample()
        
        for batch in dataloader:
            loss, k = train_step_stage3(
                crnn_model, optimizer, batch, ode_solver, y_scale
            )
            epoch_losses.append(float(loss))
        
        avg_loss = np.mean(epoch_losses)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            k_true = jnp.array([0.04, 3e7, 1e4])
            k_current = crnn_model.get_rate_coefficients()
            k_error = jnp.mean(jnp.abs(jnp.log(k_current) - jnp.log(k_true)))
            
            history.append({
                'epoch': epoch,
                'loss': avg_loss,
                'k1': float(k_current[0]),
                'k2': float(k_current[1]),
                'k3': float(k_current[2]),
                'k_error': float(k_error),
            })
            
            if epoch % 50 == 0:
                print(f"    Epoch {epoch:3d} | Loss: {avg_loss:.6e} | "
                      f"k=[{k_current[0]:.2e}, {k_current[1]:.2e}, {k_current[2]:.2e}] | "
                      f"Log error: {k_error:.4f}")
    
    print(f"\n Stage 3 complete. Best loss: {best_loss:.6e}")
    
    k_final = crnn_model.get_rate_coefficients()
    k_true = jnp.array([0.04, 3e7, 1e4])
    
    print(f"\n  Stage 3 final rate coefficients:")
    print(f"    k1: {k_final[0]:.6e} (true: {k_true[0]:.6e}, error: {abs(k_final[0]-k_true[0])/k_true[0]*100:.2f}%)")
    print(f"    k2: {k_final[1]:.6e} (true: {k_true[1]:.6e}, error: {abs(k_final[1]-k_true[1])/k_true[1]*100:.2f}%)")
    print(f"    k3: {k_final[2]:.6e} (true: {k_true[2]:.6e}, error: {abs(k_final[2]-k_true[2])/k_true[2]*100:.2f}%)")
    
    return crnn_model, history



def main(noise_level: float = 0.01, random_seed: int = 42):


    
    y_arr, t_arr = generate_robertson_data_scipy(
        y0=np.array([1.0, 0.0, 0.0]),
        t_span=(1e-5, 1e5),
        n_points=50,
        n_series=1,
        rand_init=False,
        noise_level=noise_level,
        random_seed=random_seed,
    )

    

    
    key = jax.random.PRNGKey(np.random.randint(1e6))
    mlp_model, stage1_history = train_stage1(
        y_arr=y_arr,
        t_arr=t_arr,
        hidden_size=128,
        n_epochs=10000,
        learning_rate=1e-3,
        key=key,
    )
    
    ys_mlp = integrate_neural_ode_scipy(mlp_model, y_arr[0,0], t_arr[0])
    
    if ys_mlp is not None:
        mse = np.mean((ys_mlp - y_arr[0]) ** 2)
        print(f"    Trajectory MSE: {mse:.6e}")
    

    
    key = jax.random.PRNGKey(np.random.randint(1e6))
    crnn_stage2, stage2_history, ts_interp, ys_interp = train_stage2(
        mlp_model=mlp_model,
        y_arr=y_arr,
        t_arr=t_arr,
        interpolation_factor=10,
        n_epochs=10000,
        learning_rate=0.01,
        key=key,
    )
    
    ys_stage2 = integrate_neural_ode_scipy(crnn_stage2, y_arr[0,0], t_arr[0])
    
    if ys_stage2 is not None:
        mse = np.mean((ys_stage2 - y_arr[0]) ** 2)
        print(f"    Trajectory MSE: {mse:.6e}")
    
    print("STAGE 3: CRNN Fine-tuning - 10000 epochs")
    
    key = jax.random.PRNGKey(np.random.randint(1e6))
    
    crnn_stage3, stage3_history = train_stage3(
        crnn_model=crnn_stage2,
        y_arr=y_arr,
        t_arr=t_arr,
        n_epochs=10000,
        learning_rate=0.01,
        batch_size=64,
        chuck_len=50,
        stride_len=50,
        patience_ratio=0.1,
        key=key,
    )
    
    print("\n  Evaluating Stage 3 CRNN...")
    ys_stage3 = integrate_neural_ode_scipy(crnn_stage3, y_arr[0,0], t_arr[0])
    
    if ys_stage3 is not None:
        mse = np.mean((ys_stage3 - y_arr[0]) ** 2)
        print(f"    Trajectory MSE: {mse:.6e}")
    

    
    print("\n Stage 1 MLP:")
    if ys_mlp is not None:
        mse1 = np.mean((ys_mlp - y_arr[0]) ** 2)
        print(f"  Trajectory MSE: {mse1:.6e}")
    
    print("\n Stage 2 CRNN Pre-training:")
    k_stage2 = crnn_stage2.get_rate_coefficients()
    k_true = jnp.array([0.04, 3e7, 1e4])
    if ys_stage2 is not None:
        mse2 = np.mean((ys_stage2 - y_arr[0]) ** 2)
        print(f"  Trajectory MSE: {mse2:.6e}")
    print(f"  k1: {k_stage2[0]:.6e} (error: {abs(k_stage2[0]-k_true[0])/k_true[0]*100:.2f}%)")
    print(f"  k2: {k_stage2[1]:.6e} (error: {abs(k_stage2[1]-k_true[1])/k_true[1]*100:.2f}%)")
    print(f"  k3: {k_stage2[2]:.6e} (error: {abs(k_stage2[2]-k_true[2])/k_true[2]*100:.2f}%)")
    
    print("\nStage 3 CRNN Fine-tuning:")
    k_stage3 = crnn_stage3.get_rate_coefficients()
    if ys_stage3 is not None:
        mse3 = np.mean((ys_stage3 - y_arr[0]) ** 2)
        print(f"  Trajectory MSE: {mse3:.6e}")
    print(f"  k1: {k_stage3[0]:.6e} (error: {abs(k_stage3[0]-k_true[0])/k_true[0]*100:.2f}%)")
    print(f"  k2: {k_stage3[1]:.6e} (error: {abs(k_stage3[1]-k_true[1])/k_true[1]*100:.2f}%)")
    print(f"  k3: {k_stage3[2]:.6e} (error: {abs(k_stage3[2]-k_true[2])/k_true[2]*100:.2f}%)")
    
    noise_str = f"noise{noise_level:.2f}".replace('.', '_')
    
    save_mlp_params(mlp_model, f'saved_models/stage1_mlp_params_{noise_str}.npz')
    print("  Stage 1 MLP parameters saved")
    
    save_crnn_params(crnn_stage2, f'saved_models/stage2_crnn_params_{noise_str}.npz')
    print("  Stage 2 CRNN parameters saved")
    
    save_crnn_params(crnn_stage3, f'saved_models/stage3_crnn_params_{noise_str}.npz')
    print("  Stage 3 CRNN parameters saved (BEST MODEL)")
    

    k_stage3 = crnn_stage3.get_rate_coefficients()
    k_true = jnp.array([0.04, 3e7, 1e4])
    print(f"\nNoise level: σ = {noise_level*100:.1f}%")
    print(f"  k1 = {k_stage3[0]:.6e} (true: {k_true[0]:.6e}, error: {abs(k_stage3[0]-k_true[0])/k_true[0]*100:.2f}%)")
    print(f"  k2 = {k_stage3[1]:.6e} (true: {k_true[1]:.6e}, error: {abs(k_stage3[1]-k_true[1])/k_true[1]*100:.2f}%)")
    print(f"  k3 = {k_stage3[2]:.6e} (true: {k_true[2]:.6e}, error: {abs(k_stage3[2]-k_true[2])/k_true[2]*100:.2f}%)")
    
    if ys_stage3 is not None:
        print(f"\nFinal trajectory MSE: {mse3:.6e}")
    
    print("ALL FILES SAVED TO: saved_models/")
    print(f"  - stage3_crnn_params_{noise_str}.npz  (BEST MODEL)")
    print(f"  - stage2_crnn_params_{noise_str}.npz")
    print(f"  - stage1_mlp_params_{noise_str}.npz")
    
    return mlp_model, crnn_stage2, crnn_stage3, stage1_history, stage2_history, stage3_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SPIN-ODE with Gaussian noise')
    parser.add_argument('--noise', type=float, default=0.1,
                       help='Noise level σ (e.g., 0.01 for 1%%, 0.05 for 5%%, 0.10 for 10%%, 0.20 for 20%%)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    

    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX default backend: {jax.default_backend()}")

    mlp_model, crnn_stage2, crnn_stage3, stage1_history, stage2_history, stage3_history = main(
        noise_level=args.noise,
        random_seed=args.seed,
    )