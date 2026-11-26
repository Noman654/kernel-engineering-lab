# Source on the GPU node before each session:  source env.sh
# CUDA 13.0 toolkit matches the torch cu130 build in $HOME/miniconda3.
export PATH="/usr/local/cuda-13.0/bin:$HOME/miniconda3/bin:$PATH"
# Use one GPU by default; override with CUDA_VISIBLE_DEVICES=3 source env.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "GPU $CUDA_VISIBLE_DEVICES | $(nvcc --version | tail -1) | $(python3 -c 'import torch;print("torch", torch.__version__)')"
