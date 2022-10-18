#!/bin/bash
ROOT=../../
export PYTHONPATH=$ROOT:$PYTHONPATH
if [ ! -d "./log_srun/" ];then
mkdir log_srun
fi

for cls in bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper
do 
srun --mpi=pmi2 -p$1 -n1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=4 --job-name=$cls \
python ../../tools/vis_query.py --class_name $cls > log_srun/log_$cls.txt 2>&1 &
done
