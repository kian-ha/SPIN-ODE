#!/bin/bash
#SBATCH -J neural_ode_rhs_test # Name of the job
#SBATCH -t 24:00:00 # Duration
#SBATCH -n 1 # Number of tasks
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kian.hajireza@ri.se
#SBATCH -A EUHPC_D21_054 # Account number
#SBATCH -p boost_usr_prod # Partition
#SBATCH --qos normal # Queue - check Leonardo docs for other alternatives

## Please remember to load the environment your application may need.
## And use the variable $LOCAL_SCRATCH in your batch job script 
## to access the local fast storage on each node.

echo "===== SLURM SCRIPT  ====="
cat $0
echo "==================================="

# TODO: setup python environment
#module load jax
#module load profile/deeplrn
source jax_env/bin/activate
#source venv_spinode/bin/activate
## alias python="srun venv_spinode/bin/python"


# proposed approach
## step 1: train MLP to fit nODE traj
#python train_ode.py --config configs/spin.yaml --target rober_fit
# python train_ode.py --config configs/spin.yaml --target pollu_fit
 python train_ode.py --config configs/spin.yaml --target toy_fit

## step 2: train CRNN with deriv from interpolated traj inferenced by MLP
# python train_coll.py --config configs/spin.yaml --target rober_coll
# python train_coll.py --config configs/spin.yaml --target pollu_coll
# python train_coll.py --config configs/spin.yaml --target toy_coll

## step : fine-tune on CRNN with estimated rate coefficient
# python train_ode.py --config configs/spin.yaml --target rober_tune
# python train_ode.py --config configs/spin.yaml --target pollu_tune
# python train_ode.py --config configs/spin.yaml --target toy_tune

##################################
# Baseline: directly fit CRNN on traj with ODESolver
# python train_ode.py --config configs/crnn_ode.yaml --target rober
# python train_ode.py --config configs/crnn_ode.yaml --target pollu
# python train_ode.py --config configs/crnn_ode.yaml --target toy

##################################
# Ablation
# w/o diff: train CRNN with deriv learned from MLP
# python train_coll.py --config configs/coll_mlpoutput.yaml --target rober_coll
# python train_coll.py --config configs/coll_mlpoutput.yaml --target pollu_coll
# python train_coll.py --config configs/coll_mlpoutput.yaml --target toy_coll

# w/o interpolation: train CRNN with deriv from origin traj
# python train_coll.py --config configs/coll_difforigin.yaml --target rober
# python train_coll.py --config configs/coll_difforigin.yaml --target pollu
# python train_coll.py --config configs/coll_difforigin.yaml --target toy

# w/o physical loss
## step 1: train MLP to fit nODE traj
# python train_ode.py --config configs/spin_phyloss.yaml --target rober_fit
# python train_ode.py --config configs/spin_phyloss.yaml --target pollu_fit
# python train_ode.py --config configs/spin_phyloss.yaml --target toy_fit

## step 2: train CRNN with deriv from interpolated traj inferenced by MLP
# python train_coll.py --config configs/spin_phyloss.yaml --target rober_coll
# python train_coll.py --config configs/spin_phyloss.yaml --target pollu_coll
# python train_coll.py --config configs/spin_phyloss.yaml --target toy_coll

# Stiffness

# Toy sample points
# python train_ode.py --config configs/spin_reduce.yaml --target rober_fit
# python train_ode.py --config configs/spin_reduce.yaml --target pollu_fit
# python train_ode.py --config configs/spin_reduce.yaml --target toy_fit

# python train_coll.py --config configs/spin_reduce.yaml --target rober_coll
# python train_coll.py --config configs/spin_reduce.yaml --target pollu_coll
# python train_coll.py --config configs/spin_reduce.yaml --target toy_coll
