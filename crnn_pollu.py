"""
Based on: SPIN-ODE paper (https://arxiv.org/abs/2505.05625)
Reference: Verwer (1994) - POLLU atmospheric chemistry benchmark
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




SPECIES_NAMES = [
    'NO2', 'NO', 'O3P', 'O3', 'HO2', 'HCHO', 'CO', 'ALD', 'MEO2', 'C2O3',
    'CO2', 'PAN', 'CH3O', 'HNO3', 'O1D', 'OH', 'SO2', 'SO4', 'NO3', 'N2O5'
]

# True rate coefficients (25 reactions) from Verwer (1994)
TRUE_K = np.array([
    0.350E+00,  # 01: NO2 = NO + O3P
    0.266E+02,  # 02: NO + O3 = NO2
    0.120E+05,  # 03: HO2 + NO = NO2 + OH
    0.860E-03,  # 04: HCHO = HO2 + HO2 + CO
    0.820E-03,  # 05: HCHO = CO
    0.150E+05,  # 06: HCHO + OH = HO2 + CO
    0.130E-03,  # 07: ALD = MEO2 + HO2 + CO
    0.240E+05,  # 08: ALD + OH = C2O3
    0.165E+05,  # 09: C2O3 + NO = NO2 + MEO2 + CO2
    0.900E+04,  # 10: C2O3 + NO2 = PAN
    0.220E-01,  # 11: PAN = C2O3 + NO2
    0.120E+05,  # 12: MEO2 + NO = CH3O + NO2
    0.188E+01,  # 13: CH3O = HCHO + HO2
    0.163E+05,  # 14: NO2 + OH = HNO3
    0.480E+07,  # 15: O3P = O3
    0.350E-03,  # 16: O3 = O1D
    0.175E-01,  # 17: O3 = O3P
    0.100E+09,  # 18: O1D = OH + OH
    0.444E+12,  # 19: O1D = O3P
    0.124E+04,  # 20: SO2 + OH = SO4 + HO2
    0.210E+01,  # 21: NO3 = NO
    0.578E+01,  # 22: NO3 = NO2 + O3P
    0.474E-01,  # 23: NO2 + O3 = NO3
    0.178E+04,  # 24: NO3 + NO2 = N2O5
    0.312E+01,  # 25: N2O5 = NO3 + NO2
])

# Stoichiometric matrices (20 species x 25 reactions)
# Reactants matrix
STOI_REAC = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # NO2
    [0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],  # NO
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # O3P
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0],  # O3
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # HO2
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # HCHO
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # CO
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # ALD
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # MEO2
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # C2O3
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # CO2
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # PAN
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # CH3O
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # HNO3
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],  # O1D
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # OH
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # SO2
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # SO4
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0],  # NO3
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # N2O5
])

# Products matrix
STOI_PROD = np.array([
    [0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],  # NO2
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # NO
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],  # O3P
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # O3
    [0, 0, 0, 2, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # HO2
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # HCHO
    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # CO
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # ALD
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # MEO2
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # C2O3
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # CO2
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # PAN
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # CH3O
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # HNO3
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # O1D
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],  # OH
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # SO2
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # SO4
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],  # NO3
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # N2O5
])

STOI_NET = STOI_PROD - STOI_REAC

INITIAL_CONDITIONS = np.zeros(20)
INITIAL_CONDITIONS[SPECIES_NAMES.index('NO')] = 0.2
INITIAL_CONDITIONS[SPECIES_NAMES.index('O3')] = 0.04
INITIAL_CONDITIONS[SPECIES_NAMES.index('HCHO')] = 0.1
INITIAL_CONDITIONS[SPECIES_NAMES.index('CO')] = 0.3
INITIAL_CONDITIONS[SPECIES_NAMES.index('ALD')] = 0.01
INITIAL_CONDITIONS[SPECIES_NAMES.index('SO2')] = 0.007


def pollu_ode(t, y, args=None):
    """POLLU atmospheric chemistry ODE system"""
    y = np.clip(y, 1e-30, 1e30)
    
    # Compute reaction rates using power law
    rates = TRUE_K * np.prod(y[:, None] ** STOI_REAC, axis=0)
    
    # Compute time derivatives
    dy_dt = STOI_NET @ rates
    
    return dy_dt


def generate_pollu_data_scipy(
    y0: np.ndarray = INITIAL_CONDITIONS,
    t_span: Tuple[float, float] = (0.0, 0.1),
    n_points: int = 100,
    n_series: int = 1,
    rand_init: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:

    ts = np.linspace(t_span[0], t_span[1], n_points)
    
    y_arr_list = []
    t_arr_list = []
    
    for i in range(n_series):
        if rand_init and i > 0:
            y0_perturbed = y0 * (1 + np.random.normal(0, 0.01, size=20))
            y0_perturbed = np.clip(y0_perturbed, 0, None)
        else:
            y0_perturbed = y0
        
        solution = solve_ivp(
            pollu_ode,
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



class PolluCRNN(nnx.Module):
    
    def __init__(
        self,
        k_init: jnp.ndarray = jnp.array(TRUE_K),
        *,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()
        
        self.stoi_reac = Var(jnp.array(STOI_REAC))
        self.stoi_net = Var(jnp.array(STOI_NET))
        
        self.RO2_IDX = Var(jnp.array([], dtype=jnp.int32))
        self.RO2_K_IDX = Var(jnp.array([], dtype=jnp.int32))
        
        self.log_k = nnx.Param(jnp.log(k_init))
    
    def __call__(self, t: float, y: jnp.ndarray) -> jnp.ndarray:
        y = jnp.clip(y, 1e-30, 1e30)
        
        k = jnp.exp(self.log_k.value)
        rates = k * jnp.prod(y[:, None] ** self.stoi_reac.value, axis=0)
        
        dy_dt = self.stoi_net.value @ rates
        
        return dy_dt
    
    def get_rate_coefficients(self) -> jnp.ndarray:
        return jnp.exp(self.log_k.value)



def create_ode_solver():
    return lambda model, y0, ts: diffrax.diffeqsolve(
        diffrax.ODETerm(lambda t, y, args: model(t, y)),
        diffrax.Kvaerno3(),
        t0=ts[0],
        t1=ts[-1],
        y0=y0,
        saveat=diffrax.SaveAt(ts=ts),
        dt0=None,
        adjoint=diffrax.RecursiveCheckpointAdjoint(checkpoints=8192),
        max_steps=8192,
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-7),
    ).ys


def integrate_neural_ode_scipy(model, y0, ts):
    def ode_fn(t, y):
        return np.array(model(float(t), jnp.array(y)))
    
    try:
        sol = solve_ivp(
            ode_fn,
            [ts[0], ts[-1]],
            y0,
            method='BDF',
            t_eval=ts,
            rtol=1e-6,
            atol=1e-8,
        )
        return sol.y.T
    except Exception as e:
        print(f"    Warning: ODE integration failed: {e}")
        return None




def mse_loss(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.square(pred - target))


def scaled_mse_loss(pred: jnp.ndarray, target: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    scaled_diff = (pred - target) / scale
    return jnp.mean(jnp.square(scaled_diff))




def train_stage1(
    y_arr: np.ndarray,
    t_arr: np.ndarray,
    hidden_size: int = 128,
    n_epochs: int = 3000,
    learning_rate: float = 1e-3,
    key: jax.random.PRNGKey = None,
):

    
    y_min = jnp.array(np.min(y_arr, axis=(0, 1)))
    y_max = jnp.array(np.max(y_arr, axis=(0, 1)))
    
    dy_arr_list = []
    for i in range(y_arr.shape[0]):
        dy = np.gradient(y_arr[i], t_arr[i], axis=0)
        dy_arr_list.append(dy)
    dy_arr = np.array(dy_arr_list)
    dy_scale = jnp.array(np.max(np.abs(dy_arr), axis=(0, 1)))
    

    
    n_species = y_arr.shape[2]
    rngs = nnx.Rngs(key if key is not None else jax.random.PRNGKey(42))
    
    mlp_model = NormalizedMLP(
        n_species=n_species,
        hidden_size=hidden_size,
        y_min=y_min,
        y_max=y_max,
        dy_scale=dy_scale,
        rngs=rngs,
    )
    
    ode_solver = create_ode_solver()
    
    @nnx.jit
    def loss_fn(model, y0, ts, y_target):
        y_pred = ode_solver(model, y0, ts)
        return mse_loss(y_pred, y_target)
    
    optimizer = nnx.Optimizer(mlp_model, optax.adam(learning_rate))
    
    @nnx.jit
    def train_step(model, optimizer, y0, ts, y_target):
        loss, grads = nnx.value_and_grad(loss_fn)(model, y0, ts, y_target)
        optimizer.update(grads)
        return loss
    
    history = []
    best_loss = float('inf')
    patience_counter = 0
    patience_limit = int(n_epochs * 0.1)
    
    y0 = jnp.array(y_arr[0, 0])
    ts = jnp.array(t_arr[0])
    y_target = jnp.array(y_arr[0])
    
    for epoch in range(n_epochs):
        loss = train_step(mlp_model, optimizer, y0, ts, y_target)
        
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience_limit:
            break
        
        if epoch % 100 == 0 or epoch == n_epochs - 1:
            history.append({'epoch': epoch, 'loss': float(loss)})
            print(f"    Epoch {epoch:4d} | Loss: {loss:.6e}")
    
    print(f"\n Stage 1 complete! Best loss: {best_loss:.6e}")
    
    return mlp_model, history



def train_stage2(
    mlp_model,
    y_arr: np.ndarray,
    t_arr: np.ndarray,
    interpolation_factor: int = 10,
    n_epochs: int = 10000,
    learning_rate: float = 0.01,
    key: jax.random.PRNGKey = None,
):

    print("\n[Stage 2] Pre-training CRNN on MLP-generated derivatives...")

    
    ts_orig = t_arr[0]
    ts_interp = np.linspace(ts_orig[0], ts_orig[-1], len(ts_orig) * interpolation_factor)
    
    ys_mlp = integrate_neural_ode_scipy(mlp_model, y_arr[0, 0], ts_interp)
    
    if ys_mlp is None:
        print("  Warning: MLP integration failed, using original data")
        ts_interp = ts_orig
        ys_mlp = y_arr[0]
    
    dys_interp = np.gradient(ys_mlp, ts_interp, axis=0)
    
    
    k_init = jnp.array(TRUE_K) * jnp.exp(jax.random.normal(
        key if key is not None else jax.random.PRNGKey(43), 
        shape=(25,)
    ) * 0.5)
    
    rngs = nnx.Rngs(key if key is not None else jax.random.PRNGKey(43))
    crnn_model = PolluCRNN(k_init=k_init, rngs=rngs)
    
    @nnx.jit
    def loss_fn(model, y, dy_target):
        dy_pred = jax.vmap(lambda yi: model(0.0, yi))(y)
        return jnp.mean(jnp.square(dy_pred - dy_target))
    
    lr_schedule = optax.exponential_decay(
        init_value=learning_rate,
        transition_steps=n_epochs // 10,
        decay_rate=0.95,
    )
    optimizer = nnx.Optimizer(crnn_model, optax.adam(lr_schedule))
    
    @nnx.jit
    def train_step(model, optimizer, y, dy_target):
        loss, grads = nnx.value_and_grad(loss_fn)(model, y, dy_target)
        optimizer.update(grads)
        return loss
    
    history = []
    best_loss = float('inf')
    
    y_train = jnp.array(ys_mlp)
    dy_train = jnp.array(dys_interp)
    
    for epoch in range(n_epochs):
        loss = train_step(crnn_model, optimizer, y_train, dy_train)
        
        if loss < best_loss:
            best_loss = loss
        
        if epoch % 500 == 0 or epoch == n_epochs - 1:
            k_current = crnn_model.get_rate_coefficients()
            k_error = jnp.mean(jnp.abs(jnp.log(k_current) - jnp.log(jnp.array(TRUE_K))))
            
            history.append({
                'epoch': epoch,
                'loss': float(loss),
                'k_error': float(k_error),
            })
            
            if epoch % 1000 == 0:
                print(f"    Epoch {epoch:5d} | Loss: {loss:.6e} | Log k error: {k_error:.4f}")
    
    print(f"\n  Stage 2 complete! Best loss: {best_loss:.6e}")
    
    return crnn_model, history, ts_interp, ys_mlp



def train_stage3(
    crnn_model,
    y_arr: np.ndarray,
    t_arr: np.ndarray,
    n_epochs: int = 300,
    learning_rate: float = 0.01,
    batch_size: int = 64,
    chuck_len: int = 20,
    stride_len: int = 10,
    patience_ratio: float = 0.1,
    key: jax.random.PRNGKey = None,
):


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
        drop_last=False,
    )
    
    ode_solver = create_ode_solver()
    
    y_scale = jnp.array(np.max(y_arr, axis=(0, 1)))
    
    @nnx.jit
    def loss_fn(model, batch, y_scale):
        conc = batch['conc']  # [batch, seq_len, n_spc]
        time = batch['time']  # [batch, seq_len]
        
        def single_loss(c, t):
            y_pred = ode_solver(model, c[0], t)
            return scaled_mse_loss(y_pred, c, y_scale)
        
        losses = jax.vmap(single_loss)(conc, time)
        return jnp.mean(losses)
    
    lr_schedule = optax.exponential_decay(
        init_value=learning_rate,
        transition_steps=n_epochs // 10,
        decay_rate=0.5,
    )
    optimizer = nnx.Optimizer(crnn_model, optax.adam(lr_schedule))
    
    @nnx.jit
    def train_step(model, optimizer, batch, y_scale):
        loss, grads = nnx.value_and_grad(loss_fn)(model, batch, y_scale)
        optimizer.update(grads)
        k = model.get_rate_coefficients()
        return loss, k
    
    history = []
    best_loss = float('inf')
    
    for epoch in range(n_epochs):
        epoch_losses = []
        
        dataset.rand_sample()
        
        for batch in dataloader:
            loss, k = train_step(crnn_model, optimizer, batch, y_scale)
            epoch_losses.append(float(loss))
        
        avg_loss = np.mean(epoch_losses)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            k_current = crnn_model.get_rate_coefficients()
            k_error = jnp.mean(jnp.abs(jnp.log(k_current) - jnp.log(jnp.array(TRUE_K))))
            
            history.append({
                'epoch': epoch,
                'loss': avg_loss,
                'k_error': float(k_error),
            })
            
            if epoch % 50 == 0:
                print(f"    Epoch {epoch:3d} | Loss: {avg_loss:.6e} | Log k error: {k_error:.4f}")
    
    print(f"\n  Stage 3 complete! Best loss: {best_loss:.6e}")
    
    return crnn_model, history


def save_crnn_params(crnn_model, filepath: str):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    k_values = np.array(crnn_model.get_rate_coefficients())
    log_k_values = np.array(crnn_model.log_k.value)
    
    np.savez(
        filepath,
        rate_coefficients=k_values,
        log_rate_coefficients=log_k_values,
        stoi_reac=np.array(crnn_model.stoi_reac.value),
        stoi_net=np.array(crnn_model.stoi_net.value),
        species_names=SPECIES_NAMES,
        true_k=TRUE_K,
    )
    
    print(f"  Saved CRNN parameters to: {filepath}")




def main():

    
    print("\n[0] Generating POLLU trajectory data...")
    y_arr, t_arr = generate_pollu_data_scipy(
        y0=INITIAL_CONDITIONS,
        t_span=(0.0, 0.1),
        n_points=100,
        n_series=1,
        rand_init=False,
    )

    
    print("STAGE 1: Black-box Neural ODE (MLP)")
    
    key = jax.random.PRNGKey(42)
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
    
    # Stage 2
    print("STAGE 2: CRNN Pre-training")
    
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
    
    print("\n  Evaluating Stage 2 CRNN")
    ys_stage2 = integrate_neural_ode_scipy(crnn_stage2, y_arr[0,0], t_arr[0])
    
    if ys_stage2 is not None:
        mse = np.mean((ys_stage2 - y_arr[0]) ** 2)
        print(f"    Trajectory MSE: {mse:.6e}")
    
    # Stage 3
    print("STAGE 3: CRNN Fine-tuning - 300 epochs, lr=0.01, BATCHED")
    
    key = jax.random.PRNGKey(np.random.randint(1e6))
    
    crnn_stage3, stage3_history = train_stage3(
        crnn_model=crnn_stage2,
        y_arr=y_arr,
        t_arr=t_arr,
        n_epochs=10000,
        learning_rate=0.01,
        batch_size=64,
        chuck_len=20,
        stride_len=10,
        patience_ratio=0.1,
        key=key,
    )
    
    print("\n  Evaluating Stage 3 CRNN...")
    ys_stage3 = integrate_neural_ode_scipy(crnn_stage3, y_arr[0,0], t_arr[0])
    
    if ys_stage3 is not None:
        mse = np.mean((ys_stage3 - y_arr[0]) ** 2)
        print(f"    Trajectory MSE: {mse:.6e}")

    
    print("\nStage 1 MLP:")
    if ys_mlp is not None:
        mse1 = np.mean((ys_mlp - y_arr[0]) ** 2)
        print(f"  Trajectory MSE: {mse1:.6e}")
    
    print("\nStage 2 CRNN Pre-training:")
    k_stage2 = crnn_stage2.get_rate_coefficients()
    k_error2 = jnp.mean(jnp.abs(jnp.log(k_stage2) - jnp.log(jnp.array(TRUE_K))))
    if ys_stage2 is not None:
        mse2 = np.mean((ys_stage2 - y_arr[0]) ** 2)
        print(f"  Trajectory MSE: {mse2:.6e}")
    print(f"  Mean log k error: {k_error2:.4f}")
    
    print("\nStage 3 CRNN Fine-tuning:")
    k_stage3 = crnn_stage3.get_rate_coefficients()
    k_error3 = jnp.mean(jnp.abs(jnp.log(k_stage3) - jnp.log(jnp.array(TRUE_K))))
    if ys_stage3 is not None:
        mse3 = np.mean((ys_stage3 - y_arr[0]) ** 2)
        print(f"  Trajectory MSE: {mse3:.6e}")
    print(f"  Mean log k error: {k_error3:.4f}")
    

    
    save_crnn_params(crnn_stage3, 'saved_models/stage3_pollution_params.npz')
    print(" Stage 3 CRNN parameters saved (BEST MODEL)")
    
    # Detailed rate coefficient comparison
    print("RATE COEFFICIENT COMPARISON (Stage 3)")
    print(f"\n{'Reaction':4s} {'True k':>12s} {'Predicted k':>12s} {'Rel. Error':>12s}")
    
    k_true = np.array(TRUE_K)
    k_pred = np.array(k_stage3)
    
    for i in range(len(k_true)):
        rel_error = abs(k_pred[i] - k_true[i]) / k_true[i] * 100
        print(f"{i+1:2d}   {k_true[i]:12.3e} {k_pred[i]:12.3e} {rel_error:11.2f}%")
    
    avg_rel_error = np.mean(np.abs(k_pred - k_true) / k_true * 100)
    print(f"Average relative error: {avg_rel_error:.2f}%")
    print(f"Mean log error: {k_error3:.4f}")
    
 
    print("\nFINAL MODEL SAVED TO: saved_models/stage3_pollution_params.npz")
    
    return mlp_model, crnn_stage2, crnn_stage3, stage1_history, stage2_history, stage3_history


if __name__ == "__main__":
    mlp_model, crnn_stage2, crnn_stage3, stage1_history, stage2_history, stage3_history = main()