#!/bin/bash
        
#SBATCH --job-name=jupyter_environ
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=50GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --error logs/jupyter-booster-%j.err
#SBATCH --output logs/jupyter-booster-%j.out
# Enable cuda and load the python module and enable the venv.

module purge
module load cuda/12.2
module load gcc/12.2.0-binut2.41
module load cmake/3.27.9
module load python/3.11.7
module load openmpi/5.0.9--gcc--12.2.0-cuda-12.2
module load cutensor/1.5.0.3--gcc--12.2.0
module load cudnn/8.9.7.29-12--gcc--12.2.0-cuda-12.2
module load nccl/2.22.3-1--gcc--12.2.0-cuda-12.2-spack0.22
module load gdrcopy/2.4.1-cuda-12.2

source pygpu_venv/bin/activate

# Get the worker list associated to this slurm job
worker_list=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))

# Set the first worker as the head node and get his ip
head_node=${worker_list[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# Print ssh tunnel instruction
jupyter_port=$(($RANDOM%(64511-50000+1)+50000))
jupyter_token=${USER}_${jupyter_port}
echo ===================================================
echo [INFO]: To access the Jupyter server, remember to open a ssh tunnel with: 
echo ssh -L $jupyter_port:$head_node_ip:$jupyter_port ${USER}@login07-ext.leonardo.cineca.it -N
echo then you can connect to the jupyter server at http://127.0.0.1:$jupyter_port/lab?token=$jupyter_token
echo ===================================================

# Start the head node
echo [INFO]: Starting jupyter notebook server on $head_node 

# Note that the jupyter notebook command is available only because we have enabled the venv
command="jupyter lab --ip=0.0.0.0 --port=${jupyter_port} --NotebookApp.token=${jupyter_token}"
echo [INFO]: $command
$command &

echo [INFO]: Your env is up and running.

sleep infinity



