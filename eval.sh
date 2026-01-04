experiment_name=sft100-epo2-deepseek-llm-7b-chat
step=200
source_path=./test_dataset_generate-$experiment_name-$step.parquet
target_path=./test_dataset_generate-$experiment_name-$step.jsonl

python ./parquet2jsonl.py --input $source_path --output $target_path
python ./extra_info.py --input $target_path --ex_name $experiment_name-$step

