import sys
import json
import tiktoken
from tqdm import tqdm
from pipeline import ArmoRMPipeline
from datasets import load_dataset, load_from_disk
from argparse import ArgumentParser
from collections import Counter
import torch.multiprocessing as mp
import pdb
import numpy as np
from utils import create_label, validate_answer, validate_answer_w_threshold, compute_sample_rewards
import random as random
random.seed(42)


if __name__ == "__main__":
    mp.set_start_method('spawn')

    # Initialize the argument parser to handle command-line inputs
    parser = ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="RLHFlow/ArmoRM-Llama3-8B-v0.1",
        help="Path to the pre-trained model (HuggingFace path or local folder)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="lmarena-ai/PPE-Human-Preference-V1",
        help="Path to the dataset (HuggingFace path or local folder)",
    )
    parser.add_argument(
        "--tie_threshold",
        type=float,
        default=0.0001,
        help="Path to the dataset (HuggingFace path or local folder)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../evals/datasets/ppe-human-preference-v1-ties_w_rewards_thres_",
        #default="../evals/datasets/ppe-human-preference-v1-sample_w_rewards",
        help="Path to the dataset (HuggingFace path or local folder)",
    )

    args = parser.parse_args()
    data = load_dataset(args.data_path)

    # Running ARMO to compute multi-dim rewards
    rm = ArmoRMPipeline(args.model_path, trust_remote_code=True)
    kwargs = {
        "model": rm
    }
    ths_kwargs = {
        "threshold": args.tie_threshold
    }

    data = (
        #data['test']
        #.filter(lambda x: x['winner'] not in ['model_a','model_b'])
        #.shuffle()
        #.select(range(990,1010))
        #.map(create_label)#, num_proc=8)
        #.map(compute_sample_rewards, fn_kwargs=kwargs)#, num_proc=8)
        #.map(validate_answer)#, num_proc=8)
        #.map(validate_answer_w_threshold, fn_kwargs=ths_kwargs)#, num_proc=8)
        load_from_disk(args.output_path + str(args.tie_threshold))
        )

    #accuracy = (sum(1 for x in data if x['is_correct'] is True) / len(data) * 100) if data else 0
    #accuracy_w_thres = (sum(1 for x in data if x['is_correct_w_thres'] is True) / len(data) * 100) if data else 0
    
    reward_diffs = [max(x["rewards"]) - min(x["rewards"]) for x in data]
    mean_diffs = np.mean(reward_diffs)
    median_diffs = np.median(reward_diffs)
    min_diffs = np.min(reward_diffs)
    p10 = np.percentile(reward_diffs, 0.1)
    p25 = np.percentile(reward_diffs, 0.25)
    p75 = np.percentile(reward_diffs, 0.75)
    p90 = np.percentile(reward_diffs, 0.9)
    p95 = np.percentile(reward_diffs, 0.95)
    p99 = np.percentile(reward_diffs, 0.99)
    max_diffs = np.max(reward_diffs)

    print("Number of samples: ", len(data))
    print("Mean:", mean_diffs)
    print("Median:", median_diffs)
    print("Min:", min_diffs)
    print("P10:", p10)
    print("P25:", p25)
    print("P75:", p75)
    print("P90:", p90)
    print("P95:", p95)
    print("P99:", p99)
    print("Max:", max_diffs)
    
    import matplotlib.pyplot as plt

    plt.hist(reward_diffs, bins=10, edgecolor='black')  # You can adjust the number of bins

    # Add title and labels
    plt.title('Histogram of reward differences for ties')
    plt.xlabel('Delta')
    plt.ylabel('Frequency')
    # Show the plot
    plt.show()

    #data.save_to_disk(args.output_path + str(args.tie_threshold))
