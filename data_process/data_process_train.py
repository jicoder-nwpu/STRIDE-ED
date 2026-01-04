
"""
Preprocess the empathetic dataset to parquet format
"""

import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs


def extract_answer(text):
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "CSI100E"

    dataset = datasets.load_dataset(
                "json",
                data_files={
                    'train':"./suiji5700_30000.jsonl"
                }
            )

    train_dataset = dataset["train"]

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            conversation= example.pop("input")
            response=  example.pop("response")
            output = example.pop("output")
            data = {
                "prompt": conversation,
                "response":'<thinking>{}</thinking><answer>{}</answer>'.format(output,response),
            }
            return data
                
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "./suiji5700_30000.parquet"))
    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)













       