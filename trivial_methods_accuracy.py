import pandas as pd
import numpy as np

# base_df = pd.read_csv("Statements_Arguments_Base.csv", index_col=0)
# left_df = pd.read_csv("Statements_Arguments_Left.csv", index_col=0)
# right_df = pd.read_csv("Statements_Arguments_Right.csv", index_col=0)


model_repos = [
    # "meta-llama/Llama-2-70b-chat-hf",
    # "meta-llama/Llama-2-13b-chat-hf",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Llama-3.2-3B-Instruct",
    "google/gemma-2-9b-it",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-3B-Instruct",
    "allenai/OLMo-7B-0724-Instruct-hf",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "/data/skabi9001/PoliTuned/Right-FT-Llama-3.1-8B-Instruct-DPO",
    # "/data/skabi9001/PoliTuned/Right-FT-Llama-2-7b-chat-hf-DPO",
    # "/data/skabi9001/PoliTuned/Right-FT-Llama-2-13b-chat-hf-DPO",
]
left_models = [x.split('/')[-1] for x in model_repos[:6]]
right_models = [x.split('/')[-1] for x in model_repos[6:]]



# Define model IDs
def get_gt_labels(num_left_models=len(left_models), num_right_models=len(right_models), num_rows=19):
    # Create a list of response IDs
    # response_ids = [f"Response_{i}" for i in range(1, num_rows+1)]  # Assume 10 responses for example

    # Create the first DataFrame (Models responding as "Left")
    df_left = pd.DataFrame(
        [["true_left"] * num_left_models + ["pret_left"] * num_right_models] * num_rows,
        # index=response_ids,
        columns=left_models + right_models
    )

    # Create the second DataFrame (Models responding as "Right")
    df_right = pd.DataFrame(
        [["pret_right"] * num_left_models + ["true_right"] * num_right_models] * num_rows,
        # index=response_ids,
        columns=left_models + right_models
    )
    return df_left, df_right
gt_df_left, gt_df_right = get_gt_labels()
# Print the DataFrames
# print("Models responding as Left:\n", gt_df_left)
# print("\nModels responding as Right:\n", gt_df_right)


def compute_label_accuracy(ground_truth_df, inferred_df, labels):
    """
    For each label, compute accuracy as:
      (# of correctly inferred cells for that label) / (# of cells in ground truth with that label)
    """
    accuracies = {}
    for label in labels:
        mask = (ground_truth_df == label)
        correct = (inferred_df[mask] == label).sum()
        total = mask.sum()
        accuracies[label] = correct.sum() / total.sum() if total.sum() > 0 else np.nan
    return accuracies



def calc_accuracy(df_left_pred, df_right_pred):
    # print("Left Responses by Left model Classified as True Left:", df_left_pred[left_models].stack().value_counts().to_dict()['true_left']/(len(df_left_pred)*len(left_models)))
    # print("Left Responses by Right model Classified as Pred Left:", df_left_pred[right_models].stack().value_counts().to_dict()['pret_left']/(len(df_left_pred)*len(right_models)))
    # print("Right Responses by Left model Classified as True Left:", df_right_pred[left_models].stack().value_counts().to_dict()['pret_right']/(len(df_right_pred)*len(left_models)))
    # print("Right Responses by Right model Classified as True Right:", df_right_pred[right_models].stack().value_counts().to_dict()['true_right']/(len(df_right_pred)*len(right_models)))
    
    # For "Left" responses:
    #   Left models are expected to output "true_left"
    #   Right models are expected to output "pret_left"
    accuracy_left_left = (df_left_pred[left_models] == "true_left").to_numpy().mean()
    accuracy_left_right = (df_left_pred[right_models] == "pret_left").to_numpy().mean()

    # For "Right" responses:
    #   Left models are expected to output "pret_right"
    #   Right models are expected to output "true_right"
    accuracy_right_left = (df_right_pred[left_models] == "pret_right").to_numpy().mean()
    accuracy_right_right = (df_right_pred[right_models] == "true_right").to_numpy().mean()

    # -------------------------------
    # Create a 2x2 Accuracy Summary Matrix
    # -------------------------------
    summary_matrix = pd.DataFrame({
        "Left Models": [accuracy_left_left, accuracy_right_left],
        "Right Models": [accuracy_left_right, accuracy_right_right]
    }, index=["Left Responses", "Right Responses"])

    print(summary_matrix.T)

def print_label_wise_accuracy(pred_df_left, pred_df_right):
    # Compute label-wise accuracy for left and right responses
    # Define labels for each condition
    labels_left = ["true_left", "pret_left"]
    labels_right = ["pret_right", "true_right"]
    label_accuracy_left = compute_label_accuracy(gt_df_left, pred_df_left, labels_left)
    label_accuracy_right = compute_label_accuracy(gt_df_right, pred_df_right, labels_right)

    # Combine into a single DataFrame for display
    label_accuracy_df = pd.DataFrame({
        "Left Responses": pd.Series(label_accuracy_left),
        "Right Responses": pd.Series(label_accuracy_right)
    })

    print("Label-wise Accuracy:")
    print(label_accuracy_df)

def get_judge_predictions():
    left_df = pd.read_csv("left_response_chatgpt_judge_opinion.tsv", sep='\t')
    right_df = pd.read_csv("right_response_chatgpt_judge_opinion.tsv", sep='\t')
    # models = [f"Model_{i}" for i in range(1, len(left_df.columns)+1)]
    # left_df.columns = models
    # right_df.columns = models
    return left_df, right_df


def get_ppl_predictions():
    ppl_df_left = pd.read_csv("left_response_ppl_own.tsv", sep='\t')
    ppl_df_right = pd.read_csv("right_response_ppl_own.tsv", sep='\t')
    # models = [f"Model_{i}" for i in range(1, len(left_df.columns)+1)]
    # left_df.columns = models
    # right_df.columns = models

    # print("Models responding as Left:\n", ppl_df_left)
    # print("\nModels responding as Right:\n", ppl_df_right) 
    # ------------------------------------------------
    # Infer Labels Using Corresponding Cells from the Other DF
    # ------------------------------------------------
    def infer_labels_for_df_left(ppl_left_df, ppl_right_df, left_models, right_models):
        """
        For responses as Left:
        - Left models: if ppl_left < ppl_right then "true_left", else "pret_left"
        - Right models: if ppl_left > ppl_right then "pret_left", else "true_left"
        """
        inferred = ppl_left_df.copy()
        for col in ppl_left_df.columns:
            # Compare the perplexity in the left condition to that in the right condition for the same cell
            if col in left_models:
                inferred[col] = ppl_left_df[col].combine(ppl_right_df[col],
                    lambda ppl_left, ppl_right: "true_left" if ppl_left < ppl_right else "pret_left")
            else:  # col in right_models
                inferred[col] = ppl_left_df[col].combine(ppl_right_df[col],
                    lambda ppl_left, ppl_right: "pret_left" if ppl_left > ppl_right else "true_left")
        return inferred

    def infer_labels_for_df_right(ppl_right_df, ppl_left_df, left_models, right_models):
        """
        For responses as Right:
        - Left models: if ppl_right > ppl_left then "pret_right", else "true_left"
        - Right models: if ppl_right < ppl_left then "true_right", else "pret_right"
        """
        inferred = ppl_right_df.copy()
        for col in ppl_right_df.columns:
            if col in left_models:
                inferred[col] = ppl_right_df[col].combine(ppl_left_df[col],
                    lambda ppl_right, ppl_left: "pret_right" if ppl_right > ppl_left else "true_left")
            else:  # col in right_models
                inferred[col] = ppl_right_df[col].combine(ppl_left_df[col],
                    lambda ppl_right, ppl_left: "true_right" if ppl_right < ppl_left else "pret_right")
        return inferred

    # Infer labels using corresponding cell comparisons
    left_df = infer_labels_for_df_left(ppl_df_left, ppl_df_right, left_models, right_models)
    right_df = infer_labels_for_df_right(ppl_df_right, ppl_df_left, left_models, right_models)

    # print("Models responding as Left:\n", left_df)
    # print("\nModels responding as Right:\n", right_df) 
    return left_df, right_df



def get_honesty_predictions():
    honesty_df_left = pd.read_csv("left_response_honesy.tsv", sep='\t')
    honesty_df_right = pd.read_csv("right_response_honesy.tsv", sep='\t')
    # models = [f"Model_{i}" for i in range(1, len(left_df.columns)+1)]
    # left_df.columns = models
    # right_df.columns = models

    # print("Models responding as Left:\n", honesty_df_left)
    # print("\nModels responding as Right:\n", honesty_df_right) 
    # ------------------------------------------------
    # Infer Labels Using Corresponding Cells from the Other DF
    # ------------------------------------------------
    def infer_labels_for_df_left(honesty_left_df, honesty_right_df, left_models, right_models):
        """
        For responses as Left:
        - Left models: if honesy_left > honesy_right then "true_left", else "pret_left"
        - Right models: if honesy_left < honesy_right then "pret_left", else "true_left"
        """
        inferred = honesty_left_df.copy()
        for col in honesty_left_df.columns:
            # Compare the perplexity in the left condition to that in the right condition for the same cell
            if col in left_models:
                inferred[col] = honesty_left_df[col].combine(honesty_right_df[col],
                    lambda honesty_left, honesty_right: "true_left" if honesty_left > honesty_right else "pret_left")
            else:  # col in right_models
                inferred[col] = honesty_left_df[col].combine(honesty_right_df[col],
                    lambda honesty_left, honesty_right: "pret_left" if honesty_left < honesty_right else "true_left")
        return inferred

    def infer_labels_for_df_right(honesty_right_df, honesty_left_df, left_models, right_models):
        """
        For responses as Right:
        - Left models: if honesy_right < honesy_left then "pret_right", else "true_left"
        - Right models: if honesy_right > honesy_left then "true_right", else "pret_right"
        """
        inferred = honesty_right_df.copy()
        for col in honesty_right_df.columns:
            if col in left_models:
                inferred[col] = honesty_right_df[col].combine(honesty_left_df[col],
                    lambda honesty_right, honesty_left: "pret_right" if honesty_right < honesty_left else "true_left")
            else:  # col in right_models
                inferred[col] = honesty_right_df[col].combine(honesty_left_df[col],
                    lambda honesty_right, honesty_left: "true_right" if honesty_right > honesty_left else "pret_right")
        return inferred

    # Infer labels using corresponding cell comparisons
    left_df = infer_labels_for_df_left(honesty_df_left, honesty_df_right, left_models, right_models)
    right_df = infer_labels_for_df_right(honesty_df_right, honesty_df_left, left_models, right_models)

    # print("Models responding as Left:\n", left_df)
    # print("\nModels responding as Right:\n", right_df) 
    return left_df, right_df




print("Judge:")
judge_df_left, judge_df_right = get_judge_predictions()
calc_accuracy(judge_df_left, judge_df_right)
# print_label_wise_accuracy(judge_df_left, judge_df_right)

print("\nPPL:")
ppl_df_left, ppl_df_right = get_ppl_predictions()
calc_accuracy(ppl_df_left, ppl_df_right)
# print_label_wise_accuracy(ppl_df_left, ppl_df_right)

print("\nHonesty:")
honesty_df_left, honesty_df_right = get_honesty_predictions()
calc_accuracy(honesty_df_left, honesty_df_right)
# print_label_wise_accuracy(honesty_df_left, honesty_df_right)






