CUDA_VISIBLE_DEVICES=[gpu_no] wandb agent [sweep_id]

conda activate ./venvTorch
CUDA_VISIBLE_DEVICES=5 wandb agent mresham/Segformer_and_MTL_Segformer/r84zxqv1

Sweep Agent script
./sweep_agent.sh "2 3 4" "mresham/Segformer_and_MTL_Segformer/lwn2c2ys"
