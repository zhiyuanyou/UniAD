# class_name: bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper
export PYTHONPATH=../../:$PYTHONPATH
CUDA_VISIBLE_DEVICES=$2
python -m torch.distributed.launch --nproc_per_node=$1 ../../tools/train_vis_decoder.py --class_name $3
