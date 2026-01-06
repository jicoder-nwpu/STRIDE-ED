python ./data_process_ppo.py

MODEL_PATH="./sft100-epo1-sav143-deepseek-llm-7b-ED5100fina/global_step_143/huggingface"
train_files="./train-30.parquet"
test_files="./test_dataset.parquet"
TP_VALUE=4 
INFERENCE_BATCH_SIZE=4 
GPU_MEMORY_UTILIZATION=0.7 

python3 -m verl-main.verl.trainer.main_ppo \
	data.train_files=$train_files  \
	data.val_files=$test_files  \
	data.train_batch_size=64 \
	data.max_prompt_length=1024 \
	data.max_response_length=1024 \
	actor_rollout_ref.model.path=$MODEL_PATH \
	actor_rollout_ref.actor.optim.lr=1e-6 \
	actor_rollout_ref.model.use_remove_padding=True \
	actor_rollout_ref.actor.ppo_mini_batch_size=64 \
	actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
	actor_rollout_ref.actor.fsdp_config.param_offload=True \
	actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
	actor_rollout_ref.model.enable_gradient_checkpointing=True \
	actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
	actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
	actor_rollout_ref.rollout.name=vllm \
	actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
	actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
	actor_rollout_ref.ref.fsdp_config.param_offload=True \
	critic.optim.lr=1e-5 \
	critic.model.use_remove_padding=True \
	critic.model.path=$MODEL_PATH \
	critic.model.enable_gradient_checkpointing=True \
	critic.ppo_micro_batch_size_per_gpu=16 \
	critic.model.fsdp_config.param_offload=True \
	critic.model.fsdp_config.optimizer_offload=True \
	algorithm.kl_ctrl.kl_coef=0.001 \
	custom_reward_function.path=./reward_score.py \
    custom_reward_function.name=compute_score \
	trainer.critic_warmup=0 \
	trainer.logger='["console","swanlab"]' \
	trainer.project_name='empatheticdialogue' \
	trainer.experiment_name='ppo30-epo1-sft100-epo1-sav143-deepseek-llm-7b-ED5100fina' \
	trainer.n_gpus_per_node=4 \
	trainer.nnodes=1 \
	trainer.save_freq=173 \
	trainer.test_freq=200 \
	trainer.val_before_train=False \
	trainer.total_epochs=2 $@

chmod +x ./genarate.sh
./genarate.sh