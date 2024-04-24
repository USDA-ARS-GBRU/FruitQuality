CUDA_VISIBLE_DEVICES=[gpu_no] wandb agent [sweep_id]

conda activate ./venvTorch
CUDA_VISIBLE_DEVICES=5 wandb agent mresham/Segformer_and_MTL_Segformer/qx0kxicq

CUDA_VISIBLE_DEVICES=5 wandb agent mresham/Segformer_and_MTL_Segformer/qx0kxicq --count 10

Sweep Agent script
./sweep_agent.sh "1 2 3 4 5 6 7 8 9" "mresham/Segformer_and_MTL_Segformer/o96v8ken"
