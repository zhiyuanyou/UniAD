PYTHONPATH=$PYTHONPATH:../../../ \
srun --mpi=pmi2 -p$2 -n$1 --gres=gpu:$1 --ntasks-per-node=$1 --cpus-per-task=4 --job-name=02468 \
python -u ../../../tools/train_val.py
