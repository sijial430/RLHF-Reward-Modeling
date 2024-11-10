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

import os
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

import matplotlib.pyplot as plt
plt.ion()

data = load_from_disk("../evals/datasets/ppe-human-preference-v1_w_rewards_v2")
print(data)

def filter_none(sample):
    return False if any(x == -1 for x in sample["rewards"]) else True

def sort_rewards(sample):
    top_rewards = sample["top_rewards_w_attributes"]
    sorted_rewards = []
    for rewards in top_rewards:
        sorted_rewards.append([x for x in sorted(rewards.items(), key=lambda x: x[1]['coeff'], reverse=True)])
    sample["top_rewards_w_attributes_sorted"] = json.dumps({"response_1": sorted_rewards[0],
                                                            "response_2": sorted_rewards[1]})
    return sample


data = (data
    .filter(filter_none)
    .map(sort_rewards))
#print(data[0]["top_rewards_w_attributes"][0]["code-complexity"])
#print(data[1]["top_rewards_w_attributes"][0]["code-complexity"])

#Filter: 100%|████████████| 10216/10216 [00:02<00:00, 4171.77 examples/s]
#Map: 100%|███████████████| 10160/10160 [00:05<00:00, 1974.51 examples/s]
data.save_to_disk("../evals/datasets/ppe-human-preference-v1_w_rewards_v3")


all_rewards, all_coeffs = [], []
for sample in data:
    all_rewards.extend([np.array([v["reward"] for k, v in response.items()]) for response in sample["top_rewards_w_attributes"]])
    all_coeffs.extend([np.array([v["coeff"] for k, v in response.items()]) for response in sample["top_rewards_w_attributes"]])

rewards_matrix = np.matrix(all_rewards)
coeffs_matrix = np.matrix(all_coeffs)
all_attributes = [k for k, v in data[0]["top_rewards_w_attributes"][0].items()]

#>>> rewards_matrix.var(0)
#matrix([[0.02836972, 0.02073866, 0.07292977, 0.01050571, 0.0107559 ,
#         0.00733196, 0.01109447, 0.00982971, 0.01105289, 0.02321234,
#         0.01347856, 0.0137866 , 0.02523621, 0.03788301, 0.02166984,
#         0.0178985 , 0.01808687, 0.01023662, 0.01662761]])

#>>> rewards_matrix.mean(0)
#matrix([[0.69967869, 0.69512992, 0.79617438, 0.62066407, 0.59503744,
#         0.60383665, 0.6497601 , 0.61901449, 0.79046256, 0.51323197,
#         0.78852233, 0.78600979, 0.57454063, 0.77991833, 0.73781399,
#         0.80481155, 0.79963954, 0.69483942, 0.81274131]])

#>>> coeffs_matrix.var(0)
#matrix([[3.56795936e-05, 1.68518682e-04, 8.56746390e-06, 2.31133387e-03,
#         9.59295376e-07, 1.28991100e-03, 2.76153801e-11, 4.64999972e-11,
#         3.31601885e-07, 1.98717162e-04, 1.45885089e-12, 7.23180631e-04,
#         5.59700254e-04, 1.16968105e-06, 2.92265008e-13, 2.03881608e-04,
#         4.98369010e-04, 1.93509841e-08, 3.17772899e-13]])

#>>> coeffs_matrix.mean(0)
#matrix([[ 2.35972946e-03,  3.61563329e-03,  2.68710930e-04,
#          1.18174492e-01,  8.01765625e-05,  9.59708912e-03,
#          8.44381335e-07,  2.94222810e-06,  2.30953052e-05,
#          6.07336494e-03,  3.91440023e-07,  4.16921831e-02,
#         -1.04888601e-01,  5.24503426e-05,  1.77753944e-07,
#          5.10352031e-02,  6.03349194e-02,  1.64473023e-04,
#          1.45058198e-07]])


for i in range(rewards_matrix.shape[1]):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.hist(rewards_matrix[:,i], bins=30, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'reward_hist_{all_attributes[i]}')
    ax2.hist(coeffs_matrix[:,i], bins=30, color='salmon', edgecolor='black')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'coeffs_hist_{all_attributes[i]}')
    plt.savefig(f'figures/combined_hist_{all_attributes[i]}.png')
    plt.show()

'''
attributes = [
'argilla-judge_lm', 
'argilla-overall_quality', 
'beavertails-is_safe', 
'code-complexity', 
'code-explanation', 
'code-instruction-following', 
'code-readability', 
'code-style', 
'helpsteer-coherence', 
'helpsteer-complexity', 
'helpsteer-correctness', 
'helpsteer-helpfulness', 
'helpsteer-verbosity', 
'prometheus-score', 
'ultrafeedback-helpfulness', 
'ultrafeedback-honesty', 
'ultrafeedback-instruction_following', 
'ultrafeedback-overall_score', 
'ultrafeedback-truthfulness'
]
'''