#Creates full judgement factor datasets from WildChat 50m Judgements
#Filters through all 5 models for each judgement factor and selects pairs with the highest delta

from tqdm.auto import tqdm
from datasets import Dataset, load_dataset
import numpy as np
import pandas as pd
import os
from pathlib import Path
import gc

"""Functions"""

def get_first_message_content(example):
    # Access the first message's content from the conversation
    if example['conversation'] and len(example['conversation']) > 0:
        return example['conversation'][0]['content']
    return None

def process_judgment_scores(input_string, weights=None):
    """
    Convert a string representation of scores into a weighted sum.

    Args:
        input_string (str): String representation of numpy array (e.g. '[0.1 0.2 0.3]')
        weights (array-like, optional): Weights for each position. Defaults to [1-10]

    Returns:
        float: Rounded weighted sum of scores

    Raises:
        ValueError: If input string format is invalid or dimensions don't match
        TypeError: If input is not a string
    """
    try:
        # Check input type
        if not isinstance(input_string, str):
            return -1

        # Set default weights if none provided
        if weights is None:
            weights = np.array([1,2,3,4,5,6,7,8,9,10], dtype=float)
        else:
            weights = np.array(weights, dtype=float)

        # Convert string to numpy array
        try:
            scores = np.fromstring(input_string.strip('[]'), sep=' ').astype(float)
        except:
            return -1

        # Check dimensions match
        if len(scores) != len(weights):
            return -1
            # raise ValueError(f"Score array length ({len(scores)}) does not match weights length ({len(weights)})")

        # Calculate weighted sum and round to 2 decimal places
        mean = np.round(np.sum(scores * weights), 2)

        # Variance
        second_moment = np.sum(np.power(weights - mean, 2) * scores)

        # Calculate fourth moment
        fourth_moment = np.sum(np.power(weights - mean, 4) * scores)

        # Calculate entropy (avoiding log(0))
        entropy = -np.sum(np.where(scores > 0, scores * np.log2(scores), 0))
        max_entropy = np.log2(10)  # log2(10) for 10 possible weights
        normalized_entropy = entropy / max_entropy

        return mean, np.round(second_moment, 3), round(float(fourth_moment * (1 - normalized_entropy)), 3)

    except (ValueError, TypeError) as e:
        raise type(e)(f"Error processing judgment scores: {str(e)}")

def is_unique(example):
    content = get_first_message_content(example)
    if content is None:
        return False
    if content in seen_contents:
        return False
    seen_contents.add(content)
    return True

"""Read token"""

from huggingface_hub import login

login()

"""Load datasets"""

# jdgfct="Readability"

# merged_df=pd.read_csv("/scratch/bf996/sc_data/merged_df.csv")

jdgfct="Factuality"

a = f"nyu-dice-lab/meta-llama_Llama-3.1-70B-Instruct-jdgfct-{jdgfct}"
b = f"nyu-dice-lab/nvidia_NVLM-D-72B-jdgfct-{jdgfct}"
c = f"nyu-dice-lab/Qwen_Qwen2.5-72B-Instruct-jdgfct-{jdgfct}"
d = f"nyu-dice-lab/neuralmagic_Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic-jdgfct-{jdgfct}"
e = f"nyu-dice-lab/Nexusflow_Athene-70B-jdgfct-{jdgfct}"

ds1 = load_dataset(a, split='train')
ds2 = load_dataset(b, split='train')
ds3 = load_dataset(c, split='train')
ds4 = load_dataset(d, split='train')
ds5 = load_dataset(e, split='train')

"""Process and merge datasets"""

# Create a set of unique first message contents
seen_contents = set()

# Filter the dataset to keep only rows with unique first messages
ds1 = ds1.filter(is_unique)

indices = (set(ds1['conversation_hash'])
.intersection(ds2['conversation_hash'])
.intersection(ds3['conversation_hash'])
.intersection(ds4['conversation_hash'])
.intersection(ds5['conversation_hash']))

pd_df = ds1.to_pandas()
selected_ds1 = pd_df[pd_df['conversation_hash'].isin(indices)].sort_values('conversation_hash')
print("selected ds1")
pd_df = ds2.to_pandas()
selected_ds2 = pd_df[pd_df['conversation_hash'].isin(indices)].sort_values('conversation_hash')
print("selected ds2")
pd_df = ds3.to_pandas()
selected_ds3 = pd_df[pd_df['conversation_hash'].isin(indices)].sort_values('conversation_hash')
print("selected ds3")
pd_df = ds4.to_pandas()
selected_ds4 = pd_df[pd_df['conversation_hash'].isin(indices)].sort_values('conversation_hash')
print("selected ds4")
pd_df = ds5.to_pandas()
selected_ds5 = pd_df[pd_df['conversation_hash'].isin(indices)].sort_values('conversation_hash')
print("selected ds5")
del ds1
del ds2
del ds3
del ds4
del ds5
del pd_df
gc.collect()
print("Done building selected ds1 to ds5 in Pandas")

#remove unused data for space
def clean(df):
    keys = ["content", f"judgment_meta-llama_Llama-3.1-8B-Instruct_conversation_{jdgfct}_logprob"]
    for conv in df['conversation']:
        for conv_dict in conv:
            toremove = []
            for key in conv_dict:
                if key not in keys:
                    toremove.append(key)
            for k in toremove:
                if k in conv_dict:
                    del conv_dict[k]

clean(selected_ds1)
print("cleaned ds1")
clean(selected_ds2)
print("cleaned ds2")
clean(selected_ds3)
print("cleaned ds3")
clean(selected_ds4)
print("cleaned ds4")
clean(selected_ds5)
print("cleaned ds5")

"""Create DPO"""

# Create an empty dictionary with the specified fields
dpo_dataset = {
    'system': [],
    'question': [],
    'chosen': [],
    'rejected': [],
    'chosen_score': [],
    'rejected_score': [],
}

# for row1, row2 in tqdm(zip(ds1, ds2)):
t_idx = 0

for row1, row2, row3, row4, row5 in tqdm(zip(selected_ds1.iterrows(), selected_ds2.iterrows(), selected_ds3.iterrows(),
                                                selected_ds4.iterrows(), selected_ds5.iterrows())):
    l1, l2, l3, l4 = len(dpo_dataset['system']), len(dpo_dataset['question']), len(dpo_dataset['chosen']), len(dpo_dataset['rejected'])
    assert l1 == l2 == l3 == l4, f"Lengths do not match. {l1}:{l2}:{l3}:{l4}"
    t_idx += 1
    #conversation should be a list of dicts.
    conv1 = row1[1]
    conv2 = row2[1]
    conv3 = row3[1]
    conv4 = row4[1]
    conv5 = row5[1]

    if (not(len(conv1) == len(conv2) == len(conv3) == len(conv4) == len(conv5))) or len(conv1) % 2 != 0:
        continue
    for i in range(0, len(conv1), 2):
        system = "You are a helpful assistant."
        question_until_now = "User: " + conv1[0]["content"]
        for j in range(1, i+1):
            if j % 2 == 0:
                question_until_now += "User: "
            else:
                question_until_now += "Assistant: "
            question_until_now += conv1[j]["content"]
        question = question_until_now.replace("<|begin_of_text|>", "").replace("<|END_OF_CONVERSATION|>", "").replace("<|eot_id|>", "")

        resp1 = conv1[i+1]["content"].replace("<|begin_of_text|>", "").replace("<|END_OF_CONVERSATION|>", "").replace("<|eot_id|>", "")
        #Peak Separation Index, Entropy-weighted kurtosis
        score1, var1, ewk1 = process_judgment_scores(conv1[i+1][f"judgment_meta-llama_Llama-3.1-8B-Instruct_conversation_{jdgfct}_logprob"])
        score_dict1={
            "mean" : score1,
            "variance" : var1,
            "entropy-weighted kurtosis" : ewk1,
            "response" : resp1
        }
        resp2 = conv2[i+1]["content"].replace("<|begin_of_text|>", "").replace("<|END_OF_CONVERSATION|>", "").replace("<|eot_id|>", "")
        score2, var2, ewk2 = process_judgment_scores(conv2[i+1][f"judgment_meta-llama_Llama-3.1-8B-Instruct_conversation_{jdgfct}_logprob"])
        score_dict2={
            "mean" : score2,
            "variance" : var2,
            "entropy-weighted kurtosis" : ewk2,
            "response" : resp2
        }
        resp3 = conv3[i+1]["content"].replace("<|begin_of_text|>", "").replace("<|END_OF_CONVERSATION|>", "").replace("<|eot_id|>", "")
        score3, var3, ewk3 = process_judgment_scores(conv3[i+1][f"judgment_meta-llama_Llama-3.1-8B-Instruct_conversation_{jdgfct}_logprob"])
        score_dict3={
            "mean" : score3,
            "variance" : var3,
            "entropy-weighted kurtosis" : ewk3,
            "response" : resp3
        }
        resp4 = conv4[i+1]["content"].replace("<|begin_of_text|>", "").replace("<|END_OF_CONVERSATION|>", "").replace("<|eot_id|>", "")
        score4, var4, ewk4 = process_judgment_scores(conv4[i+1][f"judgment_meta-llama_Llama-3.1-8B-Instruct_conversation_{jdgfct}_logprob"])
        score_dict4={
            "mean" : score4,
            "variance" : var4,
            "entropy-weighted kurtosis" : ewk4,
            "response" : resp4
        }
        resp5 = conv5[i+1]["content"].replace("<|begin_of_text|>", "").replace("<|END_OF_CONVERSATION|>", "").replace("<|eot_id|>", "")
        score5, var5, ewk5 = process_judgment_scores(conv5[i+1][f"judgment_meta-llama_Llama-3.1-8B-Instruct_conversation_{jdgfct}_logprob"])
        score_dict5={
            "mean" : score5,
            "variance" : var5,
            "entropy-weighted kurtosis" : ewk5,
            "response" : resp5
        }
        #Take the response pair with the highest delta
        score_dicts = [score_dict1, score_dict2, score_dict3, score_dict4, score_dict5]
        score_dicts = [score_dict in score_dicts if score_dict["mean"] > 0 else None]
        if len(score_dicts) < 2:
            continue
        else:
          min, max = None, None
          for score_dict in score_dicts:
            if min is None or score_dict["mean"] < min["mean"]:
              min = score_dict
            if max is None or score_dict["mean"] > max["mean"]:
              max = score_dict
          if abs(min["mean"] - max["mean"]) < 0.5:
            continue
          else:
            chosen = max["response"]
            chosen_score = max["mean"]
            rejected = min["response"]
            rejected_score = min["mean"]

        dpo_dataset['system'].append(system)
        dpo_dataset['question'].append(question)
        dpo_dataset['chosen'].append(chosen)
        dpo_dataset['chosen_score'].append(chosen_score)
        dpo_dataset['rejected'].append(rejected)
        dpo_dataset['rejected_score'].append(rejected_score)

"""Write token"""

login()

#Create the dataset
dpo_dataset_ds = Dataset.from_dict(dpo_dataset)

dpo_dataset_ds.push_to_hub(f"chardizard/dpo-mix5-Llama3-{jdgfct}")

#system (str), question (str), chosen (str), rejected (str)

print("Done.")
