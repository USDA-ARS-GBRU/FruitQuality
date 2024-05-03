CUDA_VISIBLE_DEVICES=[gpu_no] wandb agent [sweep_id]

conda activate ./venvTorch
CUDA_VISIBLE_DEVICES=5 wandb agent mresham/Segformer_and_MTL_Segformer/qx0kxicq

CUDA_VISIBLE_DEVICES=5 wandb agent mresham/Segformer_and_MTL_Segformer/qx0kxicq --count 10

Sweep Agent script
./sweep_agent.sh "8 7 6 4 3 2" "mresham/Cross-valid/7dhoed5t"
