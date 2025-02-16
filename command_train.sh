# example for policy training
torchrun --master_addr 192.168.3.50 --master_port 14522 --nproc_per_node 2 --nnodes 1 --node_rank 0 train.py --data_path data/collect_pens --aug --aug_jitter --num_action 20 --voxel_size 0.005 --obs_feature_dim 512 --hidden_dim 512 --nheads 8 --num_encoder_layers 4 --num_decoder_layers 1 --dim_feedforward 2048 --dropout 0.1 --ckpt_dir logs/collect_pens --batch_size 240 --num_epochs 1000 --save_epochs 50 --num_workers 24 --seed 233 

# example for data visualization & parameter check
torchrun --master_addr 192.168.3.50 --master_port 14522 --nproc_per_node 1 --nnodes 1 --node_rank 0 train.py --data_path data/collect_pens --aug --aug_jitter --num_action 20 --voxel_size 0.005 --obs_feature_dim 512 --hidden_dim 512 --nheads 8 --num_encoder_layers 4 --num_decoder_layers 1 --dim_feedforward 2048 --dropout 0.1 --ckpt_dir logs/collect_pens --batch_size 1 --num_epochs 1 --save_epochs 1 --num_workers 1 --seed 233 --vis_data
