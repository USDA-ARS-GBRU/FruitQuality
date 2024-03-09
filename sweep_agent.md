CUDA_VISIBLE_DEVICES=[gpu_no] wandb agent [sweep_id]

conda activate ./venvTorch
CUDA_VISIBLE_DEVICES=5 wandb agent mresham/Delete_Later/6wut6p4p

Sweep Agent script
./sweep_agent.sh "4 5 6" "mresham/Segformer_and_MTL_Segformer/9mc298e1"
