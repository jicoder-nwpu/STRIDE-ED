# Empathetic Dialogue Training and Evaluation

This project is a deep learning-based empathetic dialogue system training and evaluation framework that supports SFT (Supervised Fine-Tuning) and PPO (Proximal Policy Optimization) training pipelines.

## Directory Structure

```
.
├── data_process/              # Data processing scripts directory
│   ├── data_process_train.py  # Training data processing script
│   ├── data_process_test.py   # Test data processing script
│   ├── data_process_ppo.py    # PPO training data processing script
│   └── reward_score.py        # Reward function calculation script
│
├── dataset/                   # Dataset directory
│   ├── train_dataset.jsonl    # Training dataset
│   ├── test_dataset.jsonl     # Test dataset
│   └── valid_dataset.json     # Validation dataset
│
├── evaluate/                  # Evaluation scripts directory
│   ├── evaluator_bleu_dist_rouge.py  # BLEU, DIST, ROUGE evaluator
│   ├── evaluator_ppl.py              # Perplexity evaluator
│   ├── eval_bart_score_all.py        # BART Score evaluator
│   ├── eval_two_files.py             # Two-file comparison evaluation script
│   ├── eval-ppl.py                   # Perplexity evaluation script
│   ├── bart_score1.py                # BART Score calculation script
│   ├── test.json                      # Test results file
│   └── gold.json                      # Ground truth file
│
├── sft_train.sh              # SFT training script
├── ppo_train.sh              # PPO training script
├── genarate.sh               # Model generation script
└── eval.sh                   # Evaluation script
```

## Main Files Introduction

### Data Processing Scripts (`data_process/`)

- **`data_process_train.py`**: 
  - Processes training datasets, converts JSONL format to Parquet format
  - Formats dialogue data into `prompt` and `response` fields
  - Supports responses formatted with `<thinking>` and `<answer>` tags

- **`data_process_test.py`**: 
  - Processes test datasets, extracts dialogue context and answers
  - Converts test data to the format required for model inference

- **`data_process_ppo.py`**: 
  - Processes data format required for PPO training
  - Extracts additional information such as thinking process, emotion, and strategy
  - Generates data containing fields required by the reward model

- **`reward_score.py`**: 
  - Implements reward function for PPO training
  - Contains calculation logic for format reward, emotion reward, strategy reward, etc.
  - Supports identification and matching of various empathetic dialogue strategies

### Training Scripts

- **`sft_train.sh`**: 
  - Executes Supervised Fine-Tuning (SFT) training
  - Uses FSDP (Fully Sharded Data Parallel) for distributed training
  - Supports multi-GPU training configuration
  - Automatically executes generation script after training completion

- **`ppo_train.sh`**: 
  - Executes PPO reinforcement learning training
  - Configures Actor, Critic, and Reference models
  - Uses custom reward function for training
  - Automatically executes generation script after training completion

### Generation and Evaluation Scripts

- **`genarate.sh`**: 
  - Performs batch generation using trained models
  - Supports batch processing of multiple experiments and checkpoints
  - Automatically merges model weights and executes inference
  - Converts generation results to JSONL format

- **`eval.sh`**: 
  - Evaluates model generation results
  - Converts Parquet format to JSONL format
  - Extracts additional information required for evaluation

### Evaluation Tools (`evaluate/`)

- **`evaluator_bleu_dist_rouge.py`**: 
  - Implements evaluation metrics such as BLEU, DIST (diversity), ROUGE, etc.
  - Supports corpus-level and sentence-level evaluation
  - Includes F1 score calculation

- **`evaluator_ppl.py`**: 
  - Calculates model perplexity
  - Used to evaluate model's language modeling capability

- **`eval_bart_score_all.py`**: 
  - Uses BART Score for semantic similarity evaluation
  - Provides deeper semantic understanding evaluation

- **`eval_two_files.py`**: 
  - Compares evaluation results of two JSON files
  - Used for quick evaluation of differences between generation results and ground truth

## Execution Commands

### Execution Order

Execute commands in the following order to complete the full training and evaluation pipeline:

```bash
# 1. SFT Training
bash sft_train.sh

# 2. Generate Test Results
bash genarate.sh

# 3. Evaluate Generation Results
bash eval.sh
```

### Command Descriptions

#### 1. SFT Training (`bash sft_train.sh`)

This script will:
- Automatically execute `data_process_train.py` for data preprocessing
- Perform supervised fine-tuning using DeepSeek-LLM-7B-Chat as the base model
- Configure 4 GPUs for distributed training
- Automatically execute generation script after training completion

**Main Configuration Parameters**:
- `trainer.experiment_name`: Experiment name
- `trainer.total_epochs`: Number of training epochs
- `data.micro_batch_size_per_gpu`: Batch size per GPU
- `optim.lr`: Learning rate

#### 2. Generate Test Results (`bash genarate.sh`)

This script will:
- Merge model weight files
- Perform inference generation on test set using trained model
- Save generation results in Parquet format
- Automatically convert to JSONL format and extract additional information

**Main Configuration Parameters**:
- `experiment_names`: List of experiment names
- `steps`: List of checkpoint steps
- `rollout.temperature`: Generation temperature
- `rollout.top_p`: Top-p sampling parameter

#### 3. Evaluate Generation Results (`bash eval.sh`)

This script will:
- Convert generated Parquet files to JSONL format
- Extract additional information required for evaluation
- Prepare evaluation data

**Modify Before Use**:
- `experiment_name`: Experiment name
- `step`: Checkpoint step number

## Notes

1. **Data Paths**: Ensure dataset file paths are correct, especially files in the `dataset/` directory
2. **Model Paths**: Specify correct model paths in `ppo_train.sh`
3. **GPU Configuration**: Adjust GPU count and memory utilization according to actual hardware configuration
4. **Experiment Names**: Ensure experiment names are consistent across all scripts
5. **Dependencies**: Install required dependency packages such as `verl`, `transformers`, `datasets`, etc.

## Evaluation Metrics

The project supports the following evaluation metrics:
- **BLEU**: BLEU-1, BLEU-2, BLEU-3, BLEU-4
- **ROUGE**: ROUGE-1, ROUGE-2, ROUGE-L
- **DIST**: DIST-1, DIST-2 (diversity metrics)
- **F1**: Word-level F1 score
- **PPL**: Perplexity
- **BART Score**: Semantic similarity score
