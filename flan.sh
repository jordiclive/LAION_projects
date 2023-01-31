#!/bin/bash
#SBATCH --account laion
#SBATCH --partition="g80n60"
#SBATCH --job-name=flan
#SBATCH --nodes 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_%j.out

#module load intelmpi
#source /opt/intel/mpi/latest/env/vars.sh
#export LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib:/opt/amazon/efa/lib64:/usr/local/cuda-11.0/efa/lib:/usr/local/cuda-11.0/lib:/usr/local/cuda-11.0/lib64:/usr/local/cuda-11.0:/opt/nccl/build/lib:/opt/aws-ofi-nccl-install/lib:/opt/aws-ofi-nccl/lib:$LD_LIBRARY_PATH
#export NCCL_PROTO=simple
#export PATH=/opt/amazon/efa/bin:$PATH
#export LD_PRELOAD="/opt/nccl/build/lib/libnccl.so"
#
#export FI_EFA_FORK_SAFE=1
#export FI_LOG_LEVEL=1
#export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn

#export NCCL_ALGO=ring
#export NCCL_DEBUG=info
##export NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH,COLL
#
#export PYTHONFAULTHANDLER=1
#
#export CUDA_LAUNCH_BLOCKING=0
#export OMPI_MCA_mtl_base_verbose=1
#export FI_EFA_ENABLE_SHM_TRANSFER=0
#export FI_PROVIDER=efa
#export FI_EFA_TX_MIN_CREDITS=64
#export NCCL_TREE_THRESHOLD=0

#export CUDA_LAUNCH_BLOCKING=1
source /admin/home-jordiclive/jordan_flan/bin/activate
cd /admin/home-jordiclive/LAION_projects/FLAN_code
export TRANSFORMERS_CACHE=/admin/home-jordiclive/transformers_cache
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
srun --comment laion python train.py

