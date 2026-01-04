"""
Preprocess the empathetic dataset to parquet format
"""

import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs


def extract_answer(text):
    answer = text.split("<answer>")[1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    dataset = datasets.load_dataset(
                "json",
                data_files={
                    "train": "./train-30.jsonl",
                }
            )

    train_dataset = dataset["train"]
    # test_dataset = dataset["test"]

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            conversations = example.pop("conversations")
            answer_raw = example.pop("answer")
            solution = extract_answer(answer_raw)
            emotion = example.pop("emotion")
            # strategy = example.pop("strategy_sub")

            thinking = example.pop("thinking")
            thinking = thinking.split("<Strategy>")[0]

            data = {
                "data_source": "csi100e",
                "prompt": conversations,
                # "response": answer_raw,
                "response": '<thinking>{}</thinking><answer>{}</answer>'.format(thinking, solution),
                "ability": "chat",
                "reward_model": {"ground_truth": answer_raw},
                "extra_info": {
                    "split": split,
                    "answer": solution,
                    "emotion": emotion, 
                    # "strategy": strategy
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "./train-30.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
