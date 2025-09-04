#!/bin/sh -l
#SBATCH --time=48:00:00
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --cpus-per-task=48
#SBATCH --account=p200630

module load env/release/2023.1
module load Python/3.11.3-GCCcore-12.3.0
cd /home/users/u101139
source ./.bashrc
source /project/home/p200630/my_env/bin/activate

cd /home/users/u101139/ComposedImageRetrieval/

export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 4))

task="txt2img"
dataset="CIRCO"
split="test"
batch_size=32
temperature=0.1
top_p=0.9
top_k=50
n_infer_step=30
image_guidance_scale=1.5
guidance_scale=7.5
use_llm='yes'

echo "Starting task ${task} using ${dataset} ${split}"

case $task in
    "txt2img"|"txt2txt")
        log_params="${temperature}_${top_p}_${top_k}"
        ;;
    *)
        # Default case - use all parameters
        log_params="${n_infer_step}_${image_guidance_scale}_${guidance_scale}_${use_llm}_${temperature}_${top_p}_${top_k}"
        ;;
esac

CUDA_VISIBLE_DEVICES=0 python3 -u main.py --task $task --dataset $dataset --split $split --extractor CLIP --batch_size $batch_size --temperature $temperature --top_p $top_p --llm_top_k $top_k --n_infer_step $n_infer_step --image_guidance_scale $image_guidance_scale --guidance_scale $guidance_scale --use_llm $use_llm &> ./${task}_${dataset}_CLIP_${log_params}.log &

CUDA_VISIBLE_DEVICES=1 python3 -u main.py --task $task --dataset $dataset --split $split --extractor OPENCLIP --batch_size $batch_size --temperature $temperature --top_p $top_p --llm_top_k $top_k --n_infer_step $n_infer_step --image_guidance_scale $image_guidance_scale --guidance_scale $guidance_scale --use_llm $use_llm &> ./${task}_${dataset}_OPENCLIP_${log_params}.log &

CUDA_VISIBLE_DEVICES=2 python3 -u main.py --task $task --dataset $dataset --split $split --extractor OPENVISION --batch_size $batch_size --temperature $temperature --top_p $top_p --llm_top_k $top_k --n_infer_step $n_infer_step --image_guidance_scale $image_guidance_scale --guidance_scale $guidance_scale --use_llm $use_llm &> ./${task}_${dataset}_OPENVISION_${log_params}.log &

CUDA_VISIBLE_DEVICES=3 python3 -u main.py --task $task --dataset $dataset --split $split --extractor SIGLIP2 --batch_size $batch_size --temperature $temperature --top_p $top_p --llm_top_k $top_k --n_infer_step $n_infer_step --image_guidance_scale $image_guidance_scale --guidance_scale $guidance_scale --use_llm $use_llm &> ./${task}_${dataset}_SIGLIP2_${log_params}.log &

wait
echo "All tasks completed."