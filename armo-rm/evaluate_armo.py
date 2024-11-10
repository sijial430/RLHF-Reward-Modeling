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
        #default="lmarena-ai/PPE-Human-Preference-V1",
        default="../evals/datasets/ppe-human-preference-v1_w_rewards",
        help="Path to the dataset (HuggingFace path or local folder)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../evals/datasets/ppe-human-preference-v1_w_rewards_v2",
        #default="../evals/datasets/ppe-human-preference-v1-sample_w_rewards",
        help="Path to the dataset (HuggingFace path or local folder)",
    )
    parser.add_argument(
        "--output_path_hard",
        type=str,
        default="../evals/datasets/ppe-human-preference-v1-hard_w_rewards",
        #default="../evals/datasets/ppe-human-preference-v1-sample-hard_w_rewards",
        help="Path to the dataset (HuggingFace path or local folder)",
    )
    parser.add_argument(
        "--output_path_code",
        type=str,
        default="../evals/datasets/ppe-human-preference-v1-code_w_rewards",
        #default="../evals/datasets/ppe-human-preference-v1-sample-non-code_w_rewards",
        help="Path to the dataset (HuggingFace path or local folder)",
    )
    parser.add_argument(
        "--output_path_non_code",
        type=str,
        default="../evals/datasets/ppe-human-preference-v1-non-code_w_rewards",
        #default="../evals/datasets/ppe-human-preference-v1-sample-code_w_rewards",
        help="Path to the dataset (HuggingFace path or local folder)",
    )

    args = parser.parse_args()
    #data = load_dataset(args.data_path)

    '''
    # Calculate the length of each prompt
    enc = tiktoken.get_encoding("o200k_base")
    prompt_lengths = [len(enc.encode(prompt, disallowed_special=())) for prompt in data["test"]['prompt']]
    response_lengths = [len(enc.encode(response, disallowed_special=())) for response in data["test"]['response_1']]
    response_lengths += [len(enc.encode(response, disallowed_special=())) for response in data["test"]['response_2']]
    print(len(prompt_lengths), len(response_lengths))

    # Descriptive statistics for prompt length
    prompt_length_stats = {
        'mean': np.mean(prompt_lengths),
        'std': np.std(prompt_lengths),
        'min': np.min(prompt_lengths),
        '25%': np.percentile(prompt_lengths, 25), 
        '50%': np.percentile(prompt_lengths, 50),  # Median
        '75%': np.percentile(prompt_lengths, 75), 
        'max': np.max(prompt_lengths)
    }
    response_length_stats = {
        'mean': np.mean(response_lengths),
        'std': np.std(response_lengths),
        'min': np.min(response_lengths),
        '25%': np.percentile(response_lengths, 25), 
        '50%': np.percentile(response_lengths, 50),  # Median
        '75%': np.percentile(response_lengths, 75), 
        'max': np.max(response_lengths)
    }
    print("Descriptive statistics for prompt length:")
    print(prompt_length_stats)
    print("Descriptive statistics for response length:")
    print(response_length_stats)

    # Calculate distributions for binary features
    binary_features = ['winner', 'is_code', 'is_refusal', 'hard_prompt', 'easy_prompt', 'if_prompt', 'math_prompt']
    binary_distributions = {feature: dict(Counter(data["test"][feature])) for feature in binary_features}

    # Normalize distributions to get proportions
    binary_distributions_normalized = {
        feature: {k: v / sum(counts.values()) for k, v in counts.items()}
        for feature, counts in binary_distributions.items()
    }

    print("\nDistributions of winner, is_code, is_refusal, hard_prompt, easy_prompt, if_prompt, and math_prompt:")
    print(binary_distributions_normalized)
    '''

    # Running ARMO to compute multi-dim rewards
    rm = ArmoRMPipeline(args.model_path, trust_remote_code=True)
    kwargs = {
        "model": rm
    }
    ths_kwargs = {
        "threshold": 0.0001
        #TODO: test on a list of thresholds
        #"thresholds": [0.0001, 0.0005, 0.001, 0.005, 0.01]
    }

    data = (
        data['test']
        #load_from_disk(args.data_path)
        .filter(lambda x: x['winner'] in ['model_a','model_b'])
        #.shuffle()
        #.select(range(990,1010))
        .map(create_label) #, num_proc=8)
        .map(compute_sample_rewards, fn_kwargs=kwargs) #, num_proc=8)
        .map(validate_answer, num_proc=8)
        .map(validate_answer_w_threshold, fn_kwargs=ths_kwargs, num_proc=8)
        )

    accuracy = (sum(1 for x in data if x['is_correct'] is True) / len(data) * 100) if data else 0
    accuracy_w_thres = (sum(1 for x in data if x['is_correct_w_thres'] is True) / len(data) * 100) if data else 0
    
    print("Number of samples: ", len(data))
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Accuracy (w| thres = {ths_kwargs['threshold']}): {accuracy_w_thres:.3f}")

    data.save_to_disk(args.output_path)