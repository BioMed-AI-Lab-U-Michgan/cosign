#!/bin/bash

######## Build Experiment Environment ###########
exp_dir="$(readlink -f "$0")"
work_dir=$(dirname $(dirname $exp_dir))

lsun_edm_path="${work_dir}"/checkpoint/edm_bedroom256_ema.pt
lsun_cm_path="${work_dir}"/checkpoint/cd_bedroom256_lpips.pt
# For training cm on ldct only
ldct_edm_path=/path/to/ldct/edm/ckpt
ldct_edm_path=/path/to/ldct/cm/ckpt
log_path="${work_dir}"/logs

export PYTHONPATH="${work_dir}:${work_dir}/bkse"
export DATA_DIR=/path/to/output/image/folder

######## Default Values ###########
total_training_steps=50000
# for inference, batch_size should an exact divisor of size of the validation set
batch_size=4
wandb_api_key=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
wandb_user=YOUR_WANDB_USERNAME

######## Parse the Given Parameters from the Commond ###########
options=$(getopt -o t:s:n --long task:,stage:,name:,resume:,ts:,hard_constraint,control_net_ckpt: -- "$@")
eval set -- "$options"

hard_constraint=false

while true; do
  case $1 in
    # Inverse Task
    -t | --task) shift; task=$1 ; shift ;;
    # Running Stage
    -s | --stage) shift; running_stage=$1 ; shift ;;
    # Experimental Name
    -n | --name) shift; exp_name=$1 ; shift ;;
    # Resume Ckpt Path
    --resume) shift; resume=$1 ; shift ;;

    # Single/Multi-step Sampling Intervals
    --ts) shift; ts=$1 ; shift ;;
    # Hard Measurement Constraint
    --hard_constraint) hard_constraint=true ; shift ;;
    # ControlNet ckpt for sampling
    --control_net_ckpt) shift; control_net_ckpt=$1 ; shift ;;

    --) shift ; break ;;
    *) echo "Invalid option: $1" exit 1 ;;
  esac
done

### Value check ###
if [ -z "$running_stage" ]; then
    echo "[Error] Please specify the running stage"
    exit 1
fi

if [[ $running_stage == *"train"* ]]; then
    export OPENAI_LOG_FORMAT=stdout,log,csv,wandb
else
    export OPENAI_LOG_FORMAT=stdout,log,csv
fi

if [ -z "$exp_name" ]; then
    if [ -z "$task" ]; then
        exp_name=$running_stage
    else
        exp_name="${running_stage}-${task}"
    fi
fi
export OPENAI_LOGDIR="${log_path}/${exp_name}"

if [ -z "$ts" ]; then
    ts=0,39
fi

######## Configuration ###########
if [ "$running_stage" == "train_cc" ] || [ "$running_stage" == "sample_cc" ]; then
    case "$task" in
    "inpainting")
        task_config="${work_dir}"/config/inpainting_config.yaml
        ;;
    "super_resolution")
        task_config="${work_dir}"/config/super_resolution_config.yaml
        ;;
    "nonlinear_deblur")
        task_config="${work_dir}"/config/nonlinear_deblur_config.yaml
        ;;
    "ct_recon")
        task_config="${work_dir}"/config/ct_recon_config.yaml
        ;;
    *)
        echo "Unknown task: $task or task not specified"
        exit 1
        ;;
    esac
fi

if [[ $running_stage == *"train"* ]]; then
    if [ "$task" == "ct_recon" ]; then
        data_dir="${work_dir}"/datasets/ldct_train.npy
        dataset_mode=npy
        unet_path=$ldct_cm_path
        in_channels=1
        pretrained_model_path=$lsun_edm_path
        teacher_model_path=$ldct_edm_path
    else
        data_dir="${work_dir}"/datasets
        dataset_mode=lmdb
        unet_path=$lsun_cm_path
        in_channels=3
    fi
    # TODO: Define your own task and dataset here
fi

if [[ "$running_stage" == *"sample"* ]]; then
    # for cc sampling only
    if $hard_constraint; then
        sampler=ddnm
    else
        sampler=multistep
    fi

    if [ "$task" == "ct_recon" ]; then
        controlled_unet_ckpt=$ldct_cm_path
        in_channels=1
        label_root="${work_dir}"/datasets/ldct_val
    else
        controlled_unet_ckpt=$lsun_cm_path
        in_channels=3
        label_root="${work_dir}"/datasets/lsun_val
    fi
    # TODO: Define your own task and dataset here
fi

######## Training ###########
if [ "$running_stage" == "train_cc" ]; then
    python "${work_dir}"/scripts/cc_train.py \
    --loss_type recon --loss_norm lpips --log_interval 100 --save_interval 5000 \
    --global_batch_size "$batch_size" --total_training_steps "$total_training_steps" \
    --wandb_api_key "$wandb_api_key" --wandb_user "$wandb_user" --name "$exp_name" \
    --unet_path "$unet_path" \
    --task_config "$task_config" --data_dir "$data_dir" --dataset_mode "$dataset_mode" \
    --in_channels "$in_channels" --resume_checkpoint "$resume" \
    --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False \
    --dropout 0.0 --ema_rate 0.9999,0.99994,0.9999432189950708 \
    --image_size 256 --lr 0.00005 --lr_anneal_steps 0 --num_channels 256 \
    --num_head_channels 64 --num_res_blocks 2 --resblock_updown True \
    --schedule_sampler uniform --use_fp16 True --weight_decay 0.0 --weight_schedule uniform
fi

if [ "$running_stage" == "train_edm" ]; then
    python "${work_dir}"/scripts/edm_train.py \
    --log_interval 100 --save_interval 3000 \
    --data_dir "$data_dir" --dataset_mode "$dataset_mode" \
    --in_channels "$in_channels" --global_batch_size "$batch_size" \
    --wandb_api_key "$wandb_api_key" --wandb_user "$wandb_user" --name "$exp_name" \
    --pretrained_model_path "$pretrained_model_path" --resume_checkpoint "$resume" \
    --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False \
    --dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 \
    --image_size 256 --lr 0.0001 --num_channels 256 \
    --num_head_channels 64 --num_res_blocks 2 --resblock_updown True \
    --schedule_sampler lognormal --use_fp16 True --weight_decay 0.0 --weight_schedule karras
fi

if [ "$running_stage" == "train_cm" ]; then
    python "${work_dir}"/scripts/cm_train.py \
    --loss_norm lpips --log_interval 100 --save_interval 3000 \
    --teacher_model_path "$teacher_model_path" --resume_checkpoint "$resume" \
    --global_batch_size "$batch_size" --total_training_steps "$total_training_steps" \
    --wandb_api_key "$wandb_api_key" --wandb_user "$wandb_user" --name "$exp_name" \
    --data_dir "$data_dir" --dataset_mode "$dataset_mode"  --in_channels "$in_channels" \
    --training_mode consistency_distillation --sigma_max 80 --sigma_min 0.002 \
    --target_ema_mode fixed --start_ema 0.95 --scale_mode fixed --start_scales 40 \
    --lr_anneal_steps 0 \
    --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False \
    --dropout 0.0 --teacher_dropout 0.1 --ema_rate 0.9999,0.99994,0.9999432189950708 \
    --image_size 256 --lr 0.00001 --num_channels 256 \
    --num_head_channels 64 --num_res_blocks 2 --resblock_updown True \
    --schedule_sampler uniform --use_fp16 True --weight_decay 0.0 --weight_schedule uniform
fi

######## Sampling ###########
if [ "$running_stage" == "sample_cc" ]; then
    python "${work_dir}"/scripts/recon_sample.py \
    --num_samples 10000 --batch_size "$batch_size" \
    --sampler "$sampler" --ts "$ts" --task_config "$task_config" --in_channels "$in_channels" \
    --model_path "$controlled_unet_ckpt" --control_net_path $control_net_ckpt \
    --generator determ-indiv --steps 40 \
    --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False \
    --dropout 0.0 --image_size 256 --num_channels 256 --num_head_channels 64 \
    --num_res_blocks 2 --resblock_updown True --use_fp16 True --weight_schedule uniform
fi

if [ "$running_stage" == "sample_edm" ]; then
    python "${work_dir}"/scripts/image_sample.py \
    --num_samples 1800 --batch_size "$batch_size" \
    --model_path "$ldct_edm_path" --in_channels "$in_channels" \
    --training_mode edm --generator determ-indiv  --sigma_max 80 \
    --sigma_min 0.002 --s_churn 0 --steps 40 --sampler heun \
    --attention_resolutions 32,16,8  --class_cond False --dropout 0.1 --image_size 256 \
    --num_channels 256 --num_head_channels 64 --num_res_blocks 2 \
    --resblock_updown True --use_fp16 True --use_scale_shift_norm False --weight_schedule karras
fi

if [ "$running_stage" == "sample_cm" ]; then
    python "${work_dir}"/scripts/image_sample.py \
    --num_samples 1800 --batch_size "$batch_size" --model_path "$ldct_cm_path" \
    --sampler multistep --ts "$ts" --in_channels "$in_channels" \
    --generator determ-indiv --training_mode consistency_distillation --steps 40 \
    --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False \
    --dropout 0.0 --image_size 256 --num_channels 256 --num_head_channels 64 \
    --num_res_blocks 2 --resblock_updown True --use_fp16 True --weight_schedule uniform
fi

if [[ $running_stage == *"sample"* ]]; then
    # output as png
    python "${work_dir}"/evaluations/output.py
fi

if [ "$running_stage" == "sample_cc" ]; then
    # Compute PSNR, SSIM and LPIPS
    python "${work_dir}"/evaluations/compute_similarity.py \
    --task "$task" --label_root "$label_root"
fi