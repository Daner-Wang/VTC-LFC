# CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --master_port 29500 --nproc_per_node=8 --use_env ./pruning/pruning_py/pruning_deit.py \
--device cuda \
--epochs 150 \
--lr 0.0001 \
--data-set IMNET \
--num-classes 1000 \
--dist-eval \
--model deit_small_cfged_patch16_224 \
--batch-size 256 \
--teacher-model deit_small_cfged_patch16_224 \
--distillation-alpha 0.25 \
--distillation-type hard \
--warmup-epochs 0 \
--keep-qk-scale \
--finetune-only \
--output-dir  \
--data-path  \
--teacher-path  \
--resume 
