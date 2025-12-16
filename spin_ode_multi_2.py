"""
SPIN-ODE Multi-Trajectory Training 

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

jax.config.update("jax_enable_x64", True)
torch.set_default_dtype(torch.float64)

print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")
print(f"JAX default backend: {jax.default_backend()}")


def robertson_ode(t, y, args=None):
    k1, k2, k3 = 0.04, 3e7, 1e4
    A, B, C = y[0], y[1], y[2]
    
    dA = -k1 * A + k3 * B * C
    dB = k1 * A - k2 * B * B - k3 * B * C
    dC = k2 * B * B
    
    return jnp.array([dA, dB, dC])


def generate_robertson_data_scipy(
    A0_list: np.ndarray = np.array([1.0]),
    t_span: Tuple[float, float] = (1e-5, 1e5),
    n_points: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:

    ts = np.logspace(np.log10(t_span[0]), np.log10(t_span[1]), n_points)
    
    y_arr_list = []
    t_arr_list = []
    
    for A0 in A0_list:
        y0 = np.array([A0, 0.0, 0.0])
        
        solution = solve_ivp(
            lambda t, y: np.array(robertson_ode(t, y)),
            [ts[0], ts[-1]],
            y0,
            method='BDF',
            t_eval=ts,
            rtol=1e-8,
            atol=1e-10,
        )
        
        y_arr_list.append(solution.y.T)
        t_arr_list.append(ts)
    
    y_arr = np.array(y_arr_list)
    t_arr = np.array(t_arr_list)
    
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




def compute_derivatives(ys: jnp.ndarray, ts: jnp.ndarray) -> jnp.ndarray:
    dydt = jnp.zeros_like(ys)
    
    dydt = dydt.at[0].set((ys[1] - ys[0]) / (ts[1] - ts[0]))
    
    for i in range(1, len(ts) - 1):
        dt = ts[i + 1] - ts[i - 1]
        dy = ys[i + 1] - ys[i - 1]
        dydt = dydt.at[i].set(dy / dt)
    
    dydt = dydt.at[-1].set((ys[-1] - ys[-2]) / (ts[-1] - ts[-2]))
    
    return dydt



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
        
        self.log_k1 = nnx.Param(jnp.log(k_init[0]))
        self.log_k2 = nnx.Param(jnp.log(k_init[1]))
        self.log_k3 = nnx.Param(jnp.log(k_init[2]))
    
    def get_rate_coefficients(self) -> jnp.ndarray:
        return jnp.array([
            jnp.exp(self.log_k1.value),
            jnp.exp(self.log_k2.value),
            jnp.exp(self.log_k3.value),
        ])
    
    def __call__(self, t: float, y: jnp.ndarray) -> jnp.ndarray:

        A, B, C = y[0], y[1], y[2]
        
        k1 = jnp.exp(self.log_k1.value)
        k2 = jnp.exp(self.log_k2.value)
        k3 = jnp.exp(self.log_k3.value)
        
        dA = -k1 * A + k3 * B * C
        dB = k1 * A - k2 * B * B - k3 * B * C
        dC = k2 * B * B
        
        return jnp.array([dA, dB, dC])



def integrate_neural_ode_scipy(
    model: nnx.Module,
    y0: np.ndarray,
    t_eval: np.ndarray,
) -> np.ndarray:
    
    def ode_func(t, y):
        y_jax = jnp.array(y)
        dy_dt = model(t, y_jax)
        return np.array(dy_dt)
    
    try:
        solution = solve_ivp(
            ode_func,
            [t_eval[0], t_eval[-1]],
            y0,
            method='BDF',
            t_eval=t_eval,
            rtol=1e-6,
            atol=1e-8,
        )
        
        if not solution.success:
            print(f"    Warning: Integration failed: {solution.message}")
            return None
        
        return solution.y.T
    except Exception as e:
        print(f"    Error during integration: {str(e)}")
        return None


def integrate_neural_ode_diffrax(
    model: nnx.Module,
    y0: jnp.ndarray,
    t_eval: jnp.ndarray,
) -> jnp.ndarray:
    
    def ode_func(t, y, args):
        return model(t, y)
    
    term = diffrax.ODETerm(ode_func)
    solver = diffrax.Kvaerno3()
    
    saveat = diffrax.SaveAt(ts=t_eval)
    
    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=t_eval[0],
        t1=t_eval[-1],
        dt0=1e-5,
        y0=y0,
        saveat=saveat,
        stepsize_controller=diffrax.PIDController(
            rtol=1e-6, atol=1e-8
        ),
        max_steps=10000,
    )
    
    return solution.ys




def save_mlp_params(model: NormalizedMLP, filepath: str):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    params = {
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
    
    np.savez(filepath, **params)


def save_crnn_params(model: RobertsonCRNN, filepath: str):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    params = {
        'log_k1': np.array(model.log_k1.value),
        'log_k2': np.array(model.log_k2.value),
        'log_k3': np.array(model.log_k3.value),
    }
    
    np.savez(filepath, **params)


def train_stage1(
    y_arr: np.ndarray,
    t_arr: np.ndarray,
    hidden_size: int = 128,
    n_epochs: int = 10000,
    learning_rate: float = 1e-3,
    key: jax.random.PRNGKey = None,
) -> Tuple[NormalizedMLP, list]:

    print("\n[Stage 1] Training black-box Neural ODE MLP")
    
    y_min = np.min(y_arr, axis=(0, 1))
    y_max = np.max(y_arr, axis=(0, 1))
    
    dydt_list = []
    for i in range(y_arr.shape[0]):
        dydt = compute_derivatives(jnp.array(y_arr[i]), jnp.array(t_arr[i]))
        dydt_list.append(dydt)
    
    dydt_arr = np.array([np.array(d) for d in dydt_list])
    dy_scale = np.max(np.abs(dydt_arr), axis=(0, 1))
    
    
    n_species = y_arr.shape[2]
    model = NormalizedMLP(
        n_species=n_species,
        hidden_size=hidden_size,
        y_min=jnp.array(y_min),
        y_max=jnp.array(y_max),
        dy_scale=jnp.array(dy_scale),
        rngs=nnx.Rngs(key),
    )
    
    def loss_fn(model):
        total_loss = 0.0
        n_traj = y_arr.shape[0]
        
        for i in range(n_traj):
            ts = jnp.array(t_arr[i])
            ys = jnp.array(y_arr[i])
            dydt_true = jnp.array(dydt_arr[i])
            
            dydt_pred = jax.vmap(lambda t, y: model(t, y))(ts, ys)
            
            loss = jnp.mean((dydt_pred - dydt_true) ** 2)
            total_loss += loss
        
        return total_loss / n_traj
    
    optimizer = nnx.Optimizer(
        model,
        optax.adam(learning_rate),
        wrt=nnx.Param,
    )
    
    @nnx.jit
    def train_step(model, optimizer):
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)
        return loss
    
    history = []
    best_loss = float('inf')
    

    
    for epoch in range(n_epochs):
        loss = train_step(model, optimizer)
        loss_val = float(loss)
        
        if loss_val < best_loss:
            best_loss = loss_val
        
        if epoch % 100 == 0 or epoch == n_epochs - 1:
            history.append({'epoch': epoch, 'loss': loss_val})
            print(f"    Epoch {epoch:4d} | Loss: {loss_val:.6e} | Best: {best_loss:.6e}")
    
    print(f"\n  Stage 1 complete! Best loss: {best_loss:.6e}")
    
    return model, history


def train_stage2(
    mlp_model: NormalizedMLP,
    y_arr: np.ndarray,
    t_arr: np.ndarray,
    interpolation_factor: int = 10,
    n_epochs: int = 10000,
    learning_rate: float = 0.01,
    key: jax.random.PRNGKey = None,
) -> Tuple[RobertsonCRNN, list, np.ndarray, np.ndarray]:

    print("\n[Stage 2] Pre-training CRNN on interpolated trajectories...")
    
    
    ts_interp_list = []
    ys_interp_list = []
    
    for i in range(y_arr.shape[0]):
        y0 = y_arr[i, 0]
        t_orig = t_arr[i]
        y_orig = y_arr[i]
        
        t_interp = np.logspace(
            np.log10(t_orig[0]),
            np.log10(t_orig[-1]),
            len(t_orig) * interpolation_factor
        )
        
        y_interp = integrate_neural_ode_scipy(mlp_model, y0, t_interp)
        
        if y_interp is None or np.any(np.isnan(y_interp)) or np.any(np.isinf(y_interp)):
            interp_funcs = [interp1d(t_orig, y_orig[:, j], kind='cubic') 
                           for j in range(y_orig.shape[1])]
            y_interp = np.column_stack([f(t_interp) for f in interp_funcs])
        
        ts_interp_list.append(t_interp)
        ys_interp_list.append(y_interp)
    
    ts_interp = np.array(ts_interp_list)
    ys_interp = np.array(ys_interp_list)
    
    
    dy_dt_list = []
    
    for i in range(ys_interp.shape[0]):
        dy_dt = np.gradient(ys_interp[i], ts_interp[i], axis=0)
        dy_dt_list.append(dy_dt)
    
    dy_dt_interp = np.array(dy_dt_list)
    
    k_init = jnp.array([0.04, 3e7, 1e4]) * (1 + jax.random.normal(key, (3,)) * 0.1)
    crnn_model = RobertsonCRNN(k_init=k_init, rngs=nnx.Rngs(key))
    
    print(f"  Initial k: {crnn_model.get_rate_coefficients()}")
    
    def loss_fn(model):
        total_loss = 0.0
        n_traj = ys_interp.shape[0]
        
        for i in range(n_traj):
            y = jnp.array(ys_interp[i])
            dy_target = jnp.array(dy_dt_interp[i])
            
            dy_pred = jax.vmap(lambda y_: model(0.0, y_))(y)
            
            traj_loss = jnp.mean((dy_pred - dy_target) ** 2)
            total_loss += traj_loss
        
        return total_loss / n_traj
    
    optimizer = nnx.Optimizer(
        crnn_model,
        optax.adam(learning_rate),
        wrt=nnx.Param,
    )
    
    @nnx.jit
    def train_step(model, optimizer):
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)
        return loss, model.get_rate_coefficients()
    
    history = []
    best_loss = float('inf')
    
    print(f"  Training for {n_epochs} epochs...")
    
    for epoch in range(n_epochs):
        loss, k = train_step(crnn_model, optimizer)
        loss_val = float(loss)
        
        if loss_val < best_loss:
            best_loss = loss_val
        
        if epoch % 500 == 0 or epoch == n_epochs - 1:
            k_true = jnp.array([0.04, 3e7, 1e4])
            k_error = jnp.mean(jnp.abs(jnp.log(k) - jnp.log(k_true)))
            
            history.append({
                'epoch': epoch,
                'loss': loss_val,
                'k1': float(k[0]),
                'k2': float(k[1]),
                'k3': float(k[2]),
                'k_error': float(k_error),
            })
            
            print(f"    Epoch {epoch:5d} | Loss: {loss_val:.6e} | "
                  f"k=[{k[0]:.2e}, {k[1]:.2e}, {k[2]:.2e}] | "
                  f"Log error: {k_error:.4f}")
    
    print(f"\n Stage 2 complete! Best loss: {best_loss:.6e}")
    
    k_final = crnn_model.get_rate_coefficients()
    k_true = jnp.array([0.04, 3e7, 1e4])
    
    print(f"\n  Stage 2 final rate coefficients:")
    print(f"    k1: {k_final[0]:.6e} (true: {k_true[0]:.6e}, error: {abs(k_final[0]-k_true[0])/k_true[0]*100:.2f}%)")
    print(f"    k2: {k_final[1]:.6e} (true: {k_true[1]:.6e}, error: {abs(k_final[1]-k_true[1])/k_true[1]*100:.2f}%)")
    print(f"    k3: {k_final[2]:.6e} (true: {k_true[2]:.6e}, error: {abs(k_final[2]-k_true[2])/k_true[2]*100:.2f}%)")
    
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

    print("\n[Stage 3] Fine-tuning CRNN with ODE solver")
    
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
    
    def ode_func(t, y, args):
        return crnn_model(t, y)
    
    term = diffrax.ODETerm(ode_func)
    solver = diffrax.Kvaerno3()
    
    y_scale = jnp.max(jnp.abs(y_arr))
    
    def loss_fn_batch(model, batch):
        y_batch = batch['conc']
        t_batch = batch['time']
        
        def loss_single(y_seq, t_seq):
            y0 = y_seq[0]
            
            saveat = diffrax.SaveAt(ts=t_seq)
            
            solution = diffrax.diffeqsolve(
                term,
                solver,
                t0=t_seq[0],
                t1=t_seq[-1],
                dt0=(t_seq[-1] - t_seq[0]) / 100,
                y0=y0,
                saveat=saveat,
                stepsize_controller=diffrax.PIDController(
                    rtol=1e-6, atol=1e-8
                ),
                max_steps=5000,
            )
            
            y_pred = solution.ys
            loss = jnp.mean((y_pred - y_seq) ** 2) / (y_scale ** 2)
            
            return loss
        
        losses = jax.vmap(loss_single)(y_batch, t_batch)
        return jnp.mean(losses)
    
    @nnx.jit
    def train_step_stage3(model, optimizer, batch):
        """Single training step"""
        loss, grads = nnx.value_and_grad(
            lambda m: loss_fn_batch(m, batch)
        )(model)
        
        optimizer.update(grads)
        k = model.get_rate_coefficients()
        
        return loss, k
    
    optimizer = nnx.Optimizer(
        crnn_model,
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate),
            optax.scale_by_schedule(
                optax.exponential_decay(
                    init_value=1.0,
                    transition_steps=n_epochs // 10,
                    decay_rate=0.95,
                )
            ),
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
                crnn_model, optimizer, batch
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
    
    print(f"\n Stage 3 complete! Best loss: {best_loss:.6e}")
    
    k_final = crnn_model.get_rate_coefficients()
    k_true = jnp.array([0.04, 3e7, 1e4])
    
    print(f"\n  Stage 3 final rate coefficients:")
    print(f"    k1: {k_final[0]:.6e} (true: {k_true[0]:.6e}, error: {abs(k_final[0]-k_true[0])/k_true[0]*100:.2f}%)")
    print(f"    k2: {k_final[1]:.6e} (true: {k_true[1]:.6e}, error: {abs(k_final[1]-k_true[1])/k_true[1]*100:.2f}%)")
    print(f"    k3: {k_final[2]:.6e} (true: {k_true[2]:.6e}, error: {abs(k_final[2]-k_true[2])/k_true[2]*100:.2f}%)")
    
    return crnn_model, history



def evaluate_on_test_set(
    model: RobertsonCRNN,
    A0_test: np.ndarray,
    t_span: Tuple[float, float] = (1e-5, 1e5),
    n_points: int = 50,
) -> dict:

    print("\n[Evaluation] Testing on Monte Carlo initial conditions (INTERPOLATION)...")
    print(f"  Test set: {len(A0_test)} trajectories with A0 - Uniform[{A0_test.min():.2f}, {A0_test.max():.2f}]")
    
    y_test, t_test = generate_robertson_data_scipy(A0_test, t_span, n_points)
    
    mse_list = []
    k_learned = model.get_rate_coefficients()
    
    for i in range(len(A0_test)):
        y0 = y_test[i, 0]
        t_eval = t_test[i]
        y_true = y_test[i]
        
        y_pred = integrate_neural_ode_scipy(model, y0, t_eval)
        
        if y_pred is not None:
            mse = np.mean((y_pred - y_true) ** 2)
            mse_list.append(mse)
    
    results = {
        'n_test': len(A0_test),
        'A0_range': (float(A0_test.min()), float(A0_test.max())),
        'mse_mean': float(np.mean(mse_list)),
        'mse_std': float(np.std(mse_list)),
        'mse_min': float(np.min(mse_list)),
        'mse_max': float(np.max(mse_list)),
        'k_learned': [float(k) for k in k_learned],
        'k_true': [0.04, 3e7, 1e4],
    }
    
    print(f"\n  Test Results:")
    print(f"    Mean MSE: {results['mse_mean']:.6e} ± {results['mse_std']:.6e}")
    print(f"    MSE range: [{results['mse_min']:.6e}, {results['mse_max']:.6e}]")
    print(f"    Learned k: {k_learned}")
    
    return results



def main():

    A0_train = np.array([1.0, 1.5])
    y_train, t_train = generate_robertson_data_scipy(
        A0_list=A0_train,
        t_span=(1e-5, 1e5),
        n_points=50,
    )
    
    np.random.seed(np.random.randint(1e6))
    n_test = 50
    A0_test = np.random.uniform(1.0, 1.5, n_test)
    
    
    key = jax.random.PRNGKey(np.random.randint(1e6))
    mlp_model, stage1_history = train_stage1(
        y_arr=y_train,
        t_arr=t_train,
        hidden_size=128,
        n_epochs=10000,
        learning_rate=1e-3,
        key=key,
    )
    
    print("\n  Evaluating Stage 1 MLP on training set")
    for i, A0 in enumerate(A0_train):
        ys_mlp = integrate_neural_ode_scipy(mlp_model, y_train[i,0], t_train[i])
        if ys_mlp is not None:
            mse = np.mean((ys_mlp - y_train[i]) ** 2)
            print(f"    A0={A0:.1f}: MSE = {mse:.6e}")
    
    key = jax.random.PRNGKey(np.random.randint(1e6))
    crnn_stage2, stage2_history, ts_interp, ys_interp = train_stage2(
        mlp_model=mlp_model,
        y_arr=y_train,
        t_arr=t_train,
        interpolation_factor=10,
        n_epochs=10000,
        learning_rate=0.01,
        key=key,
    )
    
    print("\n  Evaluating Stage 2 CRNN on training set")
    for i, A0 in enumerate(A0_train):
        ys_stage2 = integrate_neural_ode_scipy(crnn_stage2, y_train[i,0], t_train[i])
        if ys_stage2 is not None:
            mse = np.mean((ys_stage2 - y_train[i]) ** 2)
            print(f"    A0={A0:.1f}: MSE = {mse:.6e}")
    
    print("STAGE 3: CRNN Fine-tuning")
    
    key = jax.random.PRNGKey(np.random.randint(1e6))
    crnn_stage3, stage3_history = train_stage3(
        crnn_model=crnn_stage2,
        y_arr=y_train,
        t_arr=t_train,
        n_epochs=10000,
        learning_rate=0.01,
        batch_size=64,
        chuck_len=50,
        stride_len=50,
        patience_ratio=0.1,
        key=key,
    )
    
    print("\n  Evaluating Stage 3 CRNN on training set")
    for i, A0 in enumerate(A0_train):
        ys_stage3 = integrate_neural_ode_scipy(crnn_stage3, y_train[i,0], t_train[i])
        if ys_stage3 is not None:
            mse = np.mean((ys_stage3 - y_train[i]) ** 2)
            print(f"    A0={A0:.1f}: MSE = {mse:.6e}")
    
    # NEW: Evaluate on test set
    print("\n" + "=" * 80)
    print("TEST SET EVALUATION (INTERPOLATION)")
    print("=" * 80)
    
    test_results = evaluate_on_test_set(
        model=crnn_stage3,
        A0_test=A0_test,
        t_span=(1e-5, 1e5),
        n_points=50,
    )
    

    
    save_mlp_params(mlp_model, 'saved_models/stage1_multi_A_only_mlp_params.npz')
    print("Stage 1 MLP parameters saved")
    
    save_crnn_params(crnn_stage2, 'saved_models/stage2_multi_A_only_crnn_params.npz')
    print("Stage 2 CRNN parameters saved")
    
    save_crnn_params(crnn_stage3, 'saved_models/stage3_multi_A_only_crnn_params.npz')
    print("Stage 3 CRNN parameters saved (FINAL MODEL)")

    
    k_final = crnn_stage3.get_rate_coefficients()
    k_true = jnp.array([0.04, 3e7, 1e4])
    
    print(f"  {len(A0_train)} trajectories with A0 = {A0_train}")
    
    print(f"\nLearned rate coefficients:")
    print(f"  k1 = {k_final[0]:.6e} (true: {k_true[0]:.6e}, error: {abs(k_final[0]-k_true[0])/k_true[0]*100:.2f}%)")
    print(f"  k2 = {k_final[1]:.6e} (true: {k_true[1]:.6e}, error: {abs(k_final[1]-k_true[1])/k_true[1]*100:.2f}%)")
    print(f"  k3 = {k_final[2]:.6e} (true: {k_true[2]:.6e}, error: {abs(k_final[2]-k_true[2])/k_true[2]*100:.2f}%)")
    

    print(f"  Mean MSE: {test_results['mse_mean']:.6e} ± {test_results['mse_std']:.6e}")
    

    
    return mlp_model, crnn_stage2, crnn_stage3, test_results


if __name__ == "__main__":
    mlp_model, crnn_stage2, crnn_stage3, test_results = main()