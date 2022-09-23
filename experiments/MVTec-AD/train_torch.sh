export PYTHONPATH=../../:$PYTHONPATH
python -m torch.distributed.launch --nproc_per_node=$1 ../../tools/train_val.py
