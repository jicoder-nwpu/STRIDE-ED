

data_path=./test_dataset.jsonl
experiment_names=("sft100-epo1-sav143-deepseek-llm-7b-ED1kfina")
steps=(143)

for experiment_name in "${experiment_names[@]}"; do
  for step in "${steps[@]}"; do
    echo ">>> Running experiment: $experiment_name, step: $step"2000

    save_path=./test_dataset_generate-${experiment_name}-${step}.parquet
    local_dir=./empatheticdialogue/${experiment_name}/global_step_${step}
    model_path=${local_dir}/huggingface

    
    python3 -m verl-main.verl.model_merger merge \
        --backend fsdp \
        --local_dir $local_dir \
        --target_dir $model_path

   
    
    rm -f $local_dir/*.pt

    set -x
    SECONDS=0d

    
    python3 -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=4 \
        data.path=$data_path \
        data.prompt_key=prompt \
        data.n_samples=1 \
        data.batch_size=128 \
        data.output_path=$save_path \
        model.path=$model_path \
        +model.trust_remote_code=True \
        rollout.temperature=0 \
        rollout.top_k=50 \
        rollout.dtype=float32 \
        rollout.top_p=0.7 \
        rollout.prompt_length=1024 \
        rollout.response_length=1024 \
        rollout.tensor_model_parallel_size=4 \
        rollout.gpu_memory_utilization=0.5

    echo "耗时：$SECONDS 秒"


    target_path=./test_dataset_generate-${experiment_name}-${step}.jsonl
    python ./parquet2jsonl.py --input $save_path --output $target_path
    python ./extra_info.py --input $target_path --ex_name ${experiment_name}-${step}

  done
done
