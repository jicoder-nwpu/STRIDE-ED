python ./data_process_train.py
# python ./data_process_ppo.py



torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    -m verl-main.verl.trainer.fsdp_sft_trainer \
    data.train_files=./filtered_1k_end.parquet \
    data.val_files=./filtered_1k_end.parquet \
    data.prompt_key=prompt \
    data.response_key=response \
    data.max_length=2048 \
    data.micro_batch_size_per_gpu=4 \
	optim.lr=1e-4 \
    model.partial_pretrain=./deepseek-llm-7b-chat \
    trainer.project_name=empatheticdialogue \
    trainer.experiment_name=sft100-epo1-sav143-deepseek-llm-7b-ED1kfina \
    trainer.total_epochs=1 \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=143 \
    trainer.logger='["console","swanlab"]' \

chmod +x ./genarate.sh
./genarate.sh


