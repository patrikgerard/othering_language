model_name = "llama3_only_russian"
dataset_name = "russian_majority_vote"
saved_dfs_dir = "/nas/eclairnas01/users/pgerard/otherism/test_datasets/"
test_df_file_path = "/nas/eclairnas01/users/pgerard/otherism/test_datasets/fear_language_dataset.xlsx"
# MODEL_PATH = "/nas/eclairnas01/users/pgerard/otherism/full_models/llama_3_lora_ru_with_explanation_augmented_5_epochs_with_eval"
MODEL_PATH = "/nas/eclairnas01/users/pgerard/otherism/new_models/llama3_russian_dataset"
# MODEL_PATH = "/nas/eclairnas01/users/pgerard/otherism/new_models/llama3_ukrainian_and_russian_dataset"
# MODEL_PATH = "/nas/eclairnas01/users/pgerard/otherism/new_models/llama3_ukrainian_dataset"
TSV_PATH = f"{test_df_file_path}"
OUTPUT_PATH = f"{saved_dfs_dir}{model_name}_{dataset_name}_dataset-classified.xlsx"
system_message = "You are a chatbot trained on how Russian warbloggers 'other' their enemy. Now you classify how Ukrainian warbloggers 'other' their enemy, which is often Russians. As such, you should now look at this from the perspective of pro-Ukraine, anti-Russian Ukrainian warbloggers."
system_message = "You are a chatbot trained on how Ukrainian and Russian warbloggers 'other' their enemy. Now you classify how Russian warbloggers 'other' their enemy, which is often Ukrainians. As such, you should now look at this from the perspective of pro-Russia, anti-Ukrainian Russian warbloggers."
system_message = "Classify how Ukrainian warbloggers 'other' their enemy"
system_message = "You are a chatbot trained on how Russian and Ukrainian warbloggers 'other' their enemy. Now you classify more subtle 'otherism' in United States users that are generally far-right white supremacists that dislike other races, liberals, refugees, muslims, gay people, etc. Additionally, you detect messages that aim to incite fear about these groups, often by emphasizing threats to security, culture, or values. Note whom they target in their messages and any elements that attempt to spread fear, uncertainty, or doubt about these groups."
COLUMN_TO_ANALYZE = "text"
COLUMN_TO_ANALYZE = "Original Text"
system_message = None
TSV_PATH = "/nas/eclairnas01/users/pgerard/otherism/test_datasets/new_russian_dataset_augmented_model_merged.xlsx"
TSV_PATH = "/nas/eclairnas01/users/pgerard/otherism/test_datasets/fear_language_dataset_4.xlsx"
TSV_PATH = "/nas/eclairnas01/users/pgerard/otherism/test_datasets/majority_vote_ukrainian_augmented.xlsx"
TSV_PATH = "/nas/eclairnas01/users/pgerard/otherism/test_datasets/majority_vote_russian_augmented.xlsx"
# TSV_PATH = "/nas/eclairnas01/users/pgerard/otherism/test_datasets/ukrainian_dataset.xlsx"


import itertools


from vllm import LLM, SamplingParams
import json
# from vllm import LLM, SamplingParams

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm.auto import tqdm
import torch
import ast
import time


BATCH_SIZE = 500
SAVE_EVERY_N_ROWS = 500
DEVICE = "cpu"
# Define the system message
# system_message = None
# system_message = "You are a chatbot trained on how Russian warbloggers 'other' their enemy. Now you classify how Ukrainian warbloggers 'other' their enemy, which is often Russians. As such, you should now look at this from the perspective of pro-Ukraine, anti-Russian Ukrainian warbloggers."

# !pip install --upgrade vllm triton

import re

if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"Using GPU")
from tqdm.auto import tqdm
from typing import List, Dict, Any



# Define the sampling parameters
# Define the sampling parameters to match the original pipeline
sampling_params = SamplingParams(
    max_tokens=120,  # Equivalent to max_new_tokens
    temperature=0,   # Equivalent to do_sample=False for greedy decoding
    top_p=1,         # Equivalent to no top-p sampling
    top_k=-1,         # Equivalent to considering all tokens
    logprobs=3
)

# Initialize the LLM
llm = LLM(model=MODEL_PATH, dtype="float16")


# Filter rows where any of the target columns are -1
# Function to parse the model output
def parse_model_output(output):
    try:
        # print(output)
        classifications = eval(output.split(", explanation: ")[0])
        explanation = output.split(", explanation: ")[1]
        # print(f'classifications: {classifications}')
        # print(f'explanation: {explanation}')
        return classifications, explanation
    except Exception as e:
        classifications = eval(output.split(', "explanation": ')[0])
        explanation = output.split(', "explanation": ')[1]
        return classifications, explanation
        traceback.print_exc()
        print(e)
        return parse_model_output_classifications_prefix(output)
        print(f"Error parsing model output: {e}")
        return {}, ""

def parse_model_output_classifications_prefix(output):
    try:
        # print(output)
        classifications = eval(output.split(", explanation: ")[0].split("classifications: ")[1])
        explanation = output.split(", explanation: ")[1]
        return classifications, explanation
    except Exception as e:
        classifications = eval(output.split(', "explanation": ')[0].split('classifications: ')[1])
        explanation = output.split(', "explanation": ')[1]
        print(f'Classification: {classifications}\nExplanation: {explanation}')
        return classifications, explanation
        # print(output)
        # print(f"Error parsing model output: {e}")
        return {}, ""

import traceback

def classify_test_df_with_logits(test_df, OUTPUT_PATH, BATCH_SIZE = 100,
                     SAVE_EVERY_N_ROWS = 500,
                     COLUMN_TO_ANALYZE = "Original Text",
                     system_message = None,
                     thresholds  = {
                    'Threats to Culture or Identity': 50,
                    'Threats to Survival or Physical Security': 50,
                    'Vilification/Villainization': 50,
                    'Explicit Dehumanization': 60,
                    'None': 50}
                    ):
    # The above code is a Python script with a commented-out line `explanations_list = []`. This line
    # is currently not being executed as it is commented out. It seems like the code is intended to
    # create an empty list named `explanations_list`, but it is not active in the current state.
    classifications_list = []
    explanations_list = []
    batch_index = 0
    df_classified = test_df.copy()
    total_time = 0  # To track the total time
    total_classifications = 0  # To track the number of classifications

    for i in tqdm(range(0, len(df_classified), BATCH_SIZE), desc="Processing dataset"):
        try:
            batch_index += 1
            batch_texts = df_classified[COLUMN_TO_ANALYZE][i:i + BATCH_SIZE].tolist()
            start_time = time.time()  # Start time for classification
            # generated_responses = generate_responses_vllm(batch_texts, system_message, llm, sampling_params)
            classifications, generated_responses = analyze_text(batch_texts, system_message, llm, sampling_params)
            # print(f'Length of classifications list: {len(classifications)}')
            # analyze_text(texts, system_message, llm, sampling_params)
            end_time = time.time()  # End time for classification
            batch_time = end_time - start_time
            total_time += batch_time
            total_classifications += len(generated_responses)
            # print(classifications_list)
            classifications_list.extend(classifications)
            
            for response in generated_responses:
                classifications, explanation = parse_model_output(response)
                # classifications_list.append(classifications)
                explanations_list.append(explanation)
            # print(classifications, explanation)
            # Save intermediate results every SAVE_EVERY_N_ROWS batches
            # if batch_index % SAVE_EVERY_N_ROWS == 0:
            #     df_filtered_temp = df_classified.iloc[:len(classifications_list)].copy()
            #     df_filtered_temp['classifications'] = classifications_list
            #     df_filtered_temp['explanation'] = explanations_list
            #     classification_columns = [
            #         'Threats to Culture or Identity',
            #         'Threats to Survival or Physical Security',
            #         'Vilification/Villainization',
            #         'Explicit Dehumanization',
            #         'None'
            #     ]
            #     print(f"df_filtered_temp: {df_filtered_temp}")
            #     for column in classification_columns:
            #         print(f'Comparing {df_filtered_temp[column]} to {column} -- {thresholds[column]}')
            #         df_filtered_temp[column] = df_filtered_temp['classifications'].apply(lambda x: x.get(column, 0))
                    
            #         # if float(df_filtered_temp[column]) >= thresholds[column]:
            #         #     df_filtered_temp[column] = 1
            #         # else:
            #         #     df_filtered_temp[column] = 0
                        
            #     # Update the main dataframe with the temporary dataframe values
            #     df_classified.update(df_filtered_temp)
            #     # Save the updated dataframe back to the original file
            #     df_classified.to_csv(OUTPUT_PATH, sep='\t', index=False, encoding='utf-8')
            #     print(f"Checkpoint saved at row {i}")
        except Exception as e:
            traceback.print_exc()
            print(e)
            # break
    # Ensure lists are of the same length as DataFrame
    if len(classifications_list) != len(df_classified):
        print(f"Length of classifications_list: {len(classifications_list)}, Length of DataFrame: {len(df_classified)}")
        # Handle the discrepancy
        min_length = min(len(classifications_list), len(df_classified))
        classifications_list = classifications_list[:min_length]
        explanations_list = explanations_list[:min_length]
        df_filtered = df_classified.iloc[:min_length]
    # Add the classifications and explanations to the DataFrame
    
    df_classified['classifications'] = classifications_list
    df_classified['explanation'] = explanations_list
    classification_columns = [
        'Threats to Culture or Identity',
        'Threats to Survival or Physical Security',
        'Vilification/Villainization',
        'Explicit Dehumanization',
        'None'
    ]
    for column in classification_columns:
        # classification_columns
        # print(f'Comparing {column}: {df_classified[column]} to threshold {thresholds[column]}')
        df_classified[column] = df_classified['classifications'].apply(lambda x: x.get(column, 0))
        # threshold = thresholds[column]
        # df_classified[column] = df_classified[column].apply(lambda x: 1 if x > threshold else 0).astype(int)

        # if float(df_classified[column]) >= thresholds[column]:
        #     df_classified[column] = int(1)
        # else:
        #     df_classified[column] = int(0)
    # Save the final result to a TSV file
    # Apply the thresholds to each column and convert to integers
    # df_classified.to_csv(f"{OUTPUT_PATH}_before.tsv", sep='\t', index=False, encoding='utf-8')
    for column in classification_columns:
        # print(f'Comparing {column}: {df_classified[column]} to threshold {thresholds[column]}')

        threshold = thresholds[column]
        df_classified[column] = df_classified[column].apply(lambda x: 1 if x > threshold else 0).astype(int)

    df_classified.to_excel(OUTPUT_PATH, index=False)
    print("Files have been saved successfully.")
    average_time_per_classification = total_time / total_classifications if total_classifications else 0
    print(f"Total time taken: {total_time:4f} seconds")
    print(f"Average time per classification: {average_time_per_classification:4f} seconds")
    return df_classified, total_time, average_time_per_classification




def generate_responses_vllm(input_texts, system_message, llm: LLM, sampling_params: SamplingParams):
    prompts = []
    if system_message is not None:
        for text in input_texts:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": text}
            ]
            tokenized_chat = llm.get_tokenizer().apply_chat_template(messages, tokenize=False)
            prompts.append(tokenized_chat)
    else:
        # print('No system message')
        # system_message = "You are a chatbot trained on how Ukrainian warbloggers 'other' their enemy. Now you classify how Russian warbloggers 'other' their enemy, which is often Ukrainians. As such, you should now look at this from the perspective of pro-Russia, anti-Ukrainian Russian warbloggers."
        system_message = "Classify how Russian warbloggers 'other' their enemy"
        system_message = "Classify how Ukrainian warbloggers 'other' their enemy"
        system_message = "You are a chatbot trained on how Russian warbloggers 'other' their enemy. Now you classify more subtle 'otherism' in United States users that are generally far-right white supremacists that dislike other races, liberals, refugees, muslims, gay people, etc. Additionally, you detect messages that aim to incite fear about these groups, often by emphasizing threats to security, culture, or values. Note whom they target in their messages and any elements that attempt to spread fear, uncertainty, or doubt about these groups."
        system_message = "You are a chatbot trained on how Russian and Ukrainian warbloggers 'other' their enemy. Now you classify more subtle 'otherism' in United States users that are generally far-right white supremacists that dislike other races, liberals, refugees, muslims, gay people, etc. Additionally, you detect messages that aim to incite fear about these groups, often by emphasizing threats to security, culture, or values. Note whom they target in their messages and any elements that attempt to spread fear, uncertainty, or doubt about these groups."

        for text in input_texts:
            messages = [
                # {"role": "system", "content": system_message},
                {"role": "user", "content": f'{text}'}
                # {"role": "user", "content": f'{system_message}\n{text}'}
            ]
            tokenized_chat = llm.get_tokenizer().apply_chat_template(messages, tokenize=False)
            prompts.append(tokenized_chat)
        
    # print(f'NUMBER OF PROMPTS: {len(prompts)}')
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    responses = []
    for output in outputs:
        responses.append(output)
    return responses


import math


# responses = generate_responses_vllm(texts, None, llm, sampling_params)

import math

# def logprob_to_prob(logprob):
#     return math.exp(logprob)

# def get_probabilities_at_index(logprobs, index, token_id_0, token_id_1):
#     logprobs_at_index = logprobs[index]
#     logprob_0 = logprobs_at_index.get(token_id_0)
#     logprob_1 = logprobs_at_index.get(token_id_1)
#     if logprob_0 and logprob_1:
#         prob_0 = logprob_to_prob(logprob_0.logprob)
#         prob_1 = logprob_to_prob(logprob_1.logprob)
        
#         return (prob_0, "0"), (prob_1, "1")
#     else:
#         return None, None
    
    



def logprob_to_prob(logprob):
    return math.exp(logprob)

def normalize_probs(prob_0, prob_1):
    total_prob = prob_0 + prob_1
    normalized_prob_0 = (prob_0 / total_prob) * 100
    normalized_prob_1 = (prob_1 / total_prob) * 100
    return normalized_prob_0, normalized_prob_1

def get_probabilities_at_index(logprobs, index, token_id_0, token_id_1):
    logprobs_at_index = logprobs[index]
    logprob_0 = logprobs_at_index.get(token_id_0)
    logprob_1 = logprobs_at_index.get(token_id_1)
    
    if logprob_0 and logprob_1:
        prob_0 = logprob_to_prob(logprob_0.logprob)
        prob_1 = logprob_to_prob(logprob_1.logprob)
        
        normalized_prob_0, normalized_prob_1 = normalize_probs(prob_0, prob_1)
        
        return (normalized_prob_0, "0"), (normalized_prob_1, "1")
    else:
        return None, None


def parse_model_output(output):
    try:
        # print(output)
        classifications = eval(output.split(", explanation: ")[0])
        explanation = output.split(", explanation: ")[1]
        # print(f'classifications: {classifications}')
        # print(f'explanation: {explanation}')
        return classifications, explanation
    except Exception as e:
        classifications = eval(output.split(', "explanation": ')[0])
        explanation = output.split(', "explanation": ')[1]
        return classifications, explanation
        traceback.print_exc()
        print(e)
        return parse_model_output_classifications_prefix(output)
        print(f"Error parsing model output: {e}")
        return {}, ""

def parse_model_output_classifications_prefix(output):
    try:
        # print(output)
        classifications = eval(output.split(", explanation: ")[0].split("classifications: ")[1])
        explanation = output.split(", explanation: ")[1]
        return classifications, explanation
    except Exception as e:
        classifications = eval(output.split(', "explanation": ')[0].split('classifications: ')[1])
        explanation = output.split(', "explanation": ')[1]
        print(f'Classification: {classifications}\nExplanation: {explanation}')
        return classifications, explanation
        # print(output)
        # print(f"Error parsing model output: {e}")
        return {}, ""


def analyze_text(texts, system_message, llm, sampling_params):
    # Generate the output
    outputs = generate_responses_vllm(texts, system_message, llm, sampling_params)
    # Tokenize "0" and "1"
    tokenizer = llm.get_tokenizer()
    token_id_0 = tokenizer.encode("0")[-1]
    token_id_1 = tokenizer.encode("1")[-1]
    # print(f'Token id for 0: {token_id_0}')
    # print(f'Token id for 1: {token_id_1}')
    # Map found indices to the classification columns
    classification_columns = [
        'Threats to Culture or Identity',
        'Threats to Survival or Physical Security',
        'Vilification/Villainization',
        'Explicit Dehumanization',
        'None'
    ]
    # Print the generated text
    all_classifications_dict = []
    all_responses = []
    for output in outputs:
        try:
            token_ids = output.outputs[0].token_ids
            # print(token_ids)
            logprobs = output.outputs[0].logprobs
            # print(token_ids)
            all_responses.append(output.outputs[0].text)
            # print(all_responses)
            # Find the first 4 occurrences of either "0" or "1"
            indices = []
            for idx, token_id in enumerate(token_ids):
                if token_id in [token_id_0, token_id_1] and len(indices) < 5:
                    indices.append(idx)
            # print(f'Indices: {indices}')
            # Ensure we found exactly 4 indices

            # if len(indices) != 5:
            #     raise ValueError("Could not find exactly 5 occurrences of '0' or '1' in the token_ids.")
            # print(output.outputs[0].text)
            # Iterate over the found indices and get probabilities for "0" and "1"
            # classifications = {}
            classifications = {'Threats to Culture or Identity': 0, 
                'Threats to Survival or Physical Security': 0, 'Vilification/Villainization': 0, 
                'Explicit Dehumanization': 0, 'None': 0, 
                }
            # print(f'Indices: {indices}')
            for list_index, idx in enumerate(indices):
                prob_0, prob_1 = get_probabilities_at_index(logprobs, idx, token_id_0, token_id_1)
                # In case prob1 doesn't show up in the top 2 logits
                try:
                    classifications[classification_columns[list_index]] = prob_1[0]
                except:
                    try:
                        classifications[classification_columns[list_index]] = 0.
                    except:
                        pass
                
                # print(classifications)
                
                # if prob_0 and prob_1:
                #     print(f"{classification_columns[list_index]}:")
                #     print(f"1: {prob_1[1]}, Probability: {prob_1[0]:.3f}")
                #     print(f"0: {prob_0[1]}, Probability: {prob_0[0]:.3f}")
                # else:
                #     print(f"Token '0' or '1' not found at index {idx}.")
            # print(classifications)
            all_classifications_dict.append(classifications)
        except Exception as e:
            print(e)
            print(prob_1)
    # print(all_responses)
    return all_classifications_dict, all_responses


import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report, cohen_kappa_score

classification_columns = [
    'Threats to Culture or Identity',
    'Threats to Survival or Physical Security',
    'Vilification/Villainization',
    'Explicit Dehumanization',
    'None'
]

def classify_and_get_metrics(test_df, OUTPUT_PATH, classification_columns, COLUMN_TO_ANALYZE, system_message=None, 
                             thresholds  = {
                    'Threats to Culture or Identity': 50,
                    'Threats to Survival or Physical Security': 50,
                    'Vilification/Villainization': 50,
                    'Explicit Dehumanization': 60,
                    'None': 50}
                    ):
    classified_df, total_time, avg_time_per_classification = classify_test_df_with_logits(test_df, OUTPUT_PATH, COLUMN_TO_ANALYZE=COLUMN_TO_ANALYZE, system_message=system_message, thresholds=thresholds)

    # Extract true and predicted values
    true_values = test_df[classification_columns]
    predicted_values = classified_df[classification_columns]

    # Calculate overall metrics
    accuracy = accuracy_score(true_values, predicted_values)
    f1 = f1_score(true_values, predicted_values, average='weighted')
    precision = precision_score(true_values, predicted_values, average='weighted')
    recall = recall_score(true_values, predicted_values, average='weighted')
    conf_matrix = confusion_matrix(true_values.values.argmax(axis=1), predicted_values.values.argmax(axis=1))

    # Print overall metrics
    print()
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # Decode the labels for the kappa score
    true_values_decoded = true_values.idxmax(axis=1)
    predicted_values_decoded = predicted_values.idxmax(axis=1)

    # Calculate Cohen's Kappa for the overall classification
    overall_kappa = cohen_kappa_score(true_values_decoded, predicted_values_decoded)
    print(f"Overall Cohen's Kappa: {overall_kappa:.4f}")
    
    # Store overall metrics in a dictionary
    metrics_dict = {
        "Overall": {
            "Accuracy": round(accuracy, 5),
            "F1 Score": round(f1, 5),
            "Precision": round(precision, 5),
            "Recall": round(recall, 5),
            "Cohen's Kappa": round(overall_kappa, 5)
        },
        "Classes": {}
    }

    # Per-class metrics
    for class_column in classification_columns:
        class_true_values = true_values[class_column]
        class_predicted_values = predicted_values[class_column]
        class_accuracy = accuracy_score(class_true_values, class_predicted_values)
        class_f1 = f1_score(class_true_values, class_predicted_values)
        class_precision = precision_score(class_true_values, class_predicted_values)
        class_recall = recall_score(class_true_values, class_predicted_values)

        # Calculate Cohen's Kappa for each class
        class_kappa = cohen_kappa_score(class_true_values, class_predicted_values)

        # Count the number of actual and predicted 1s
        actual_ones = class_true_values.sum()
        predicted_ones = class_predicted_values.sum()

        print(f'--------------------  {class_column}  --------------------')
        print(f"Accuracy: {class_accuracy:.4f}")
        print(f"F1 Score: {class_f1:.4f}")
        print(f"Precision: {class_precision:.4f}")
        print(f"Recall: {class_recall:.4f}")
        print(f"Cohen's Kappa: {class_kappa:.4f}")
        print(f"Number of actual 1s: {actual_ones}")
        print(f"Number of predicted 1s: {predicted_ones}")
        print()

        metrics_dict["Classes"][class_column] = {
            "Accuracy": round(class_accuracy, 5),
            "F1 Score": round(class_f1, 5),
            "Precision": round(class_precision, 5),
            "Recall": round(class_recall, 5),
            "Cohen's Kappa": round(class_kappa, 5),
            "Number of actual 1s": int(actual_ones),
            "Number of predicted 1s": int(predicted_ones)
        }
        
    return classified_df, metrics_dict

def run_classification_multiple_times(test_df, OUTPUT_PATH, classification_columns, n, COLUMN_TO_ANALYZE):
    metrics_list = []

    for _ in range(n):
        _, stats_dict = classify_and_get_metrics(test_df, OUTPUT_PATH, classification_columns, COLUMN_TO_ANALYZE=COLUMN_TO_ANALYZE)
        metrics_list.append(stats_dict)
    
    # Convert to DataFrame for easier calculation
    metrics_df = pd.DataFrame(metrics_list)
    
    # Calculate average and standard deviation
    metrics_avg = metrics_df.mean()
    metrics_std = metrics_df.std()

    return metrics_avg, metrics_std, metrics_df

def plot_metrics(metrics_avg, metrics_std):
    # Drop confusion matrix from the metrics as it's not suitable for average/stddev calculation
    metrics_avg = metrics_avg.drop('confusion_matrix')
    metrics_std = metrics_std.drop('confusion_matrix')
    
    plt.figure(figsize=(12, 6))
    plt.errorbar(metrics_avg.index, metrics_avg, yerr=metrics_std, fmt='o', capsize=5, linestyle='None', marker='o')
    plt.title('Average Metrics with Standard Deviation')
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.grid(True)
    plt.show()


# TSV_PATH = "/nas/eclairnas01/users/pgerard/otherism/test_datasets/russian_dataset.xlsx"
COLUMN_TO_ANALYZE = 'input'
COLUMN_TO_ANALYZE = 'text'
COLUMN_TO_ANALYZE = 'Original Text'
# Load tsv file
if ".tsv" in TSV_PATH:
    test_df = pd.read_csv(TSV_PATH, sep='\t', encoding='utf-8')
else:
    test_df = pd.read_excel(TSV_PATH)
    
    
    
test_df[COLUMN_TO_ANALYZE] = test_df[COLUMN_TO_ANALYZE].astype(str)

# Filter rows where any of the target columns are -1
classification_columns = [
    'Threats to Culture or Identity',
    'Threats to Survival or Physical Security',
    'Vilification/Villainization',
    'Explicit Dehumanization',
    'None'
]

# system_message = None

thresholds  = {
                    'Threats to Culture or Identity': 50,
                    'Threats to Survival or Physical Security': 60,
                    'Vilification/Villainization': 50,
                    'Explicit Dehumanization': 50,
                    'None': 50}



classified_df, stats_dict = classify_and_get_metrics(test_df, OUTPUT_PATH, 
                                                     classification_columns,
                                                     COLUMN_TO_ANALYZE=COLUMN_TO_ANALYZE,
                                                     system_message=system_message,
                                                     thresholds=thresholds
                                                     )


print(classified_df)
print(stats_dict)



