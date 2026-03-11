import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_metrics(df):
    """
    Computes accuracy, precision, recall, and F1 for each model.
    """
    # Filter only valid answers
    valid_df = df[df['is_valid'] == True]
    
    results = []
    
    for model in df['model'].unique():
        model_data = valid_df[valid_df['model'] == model]
        
        y_true = model_data['correct_answer']
        y_pred = model_data['parsed_answer']
        
        # Calculate metrics (Macro average is better for benchmarks like MMLU)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', labels=['A', 'B', 'C', 'D'], zero_division=0
        )
        accuracy = accuracy_score(y_true, y_pred)
        
        results.append({
            "Model": model,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        })
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Load your results
    df = pd.read_csv("results/raw_results.csv")
    
    # Ensure correct/valid columns exist
    df['is_valid'] = df['parsed_answer'].isin(['A', 'B', 'C', 'D'])
    
    metrics_df = calculate_metrics(df)
    print("\n--- Model Performance Summary ---")
    print(metrics_df.to_string(index=False))
    
    # Save the summary
    metrics_df.to_csv("results/final_evaluation.csv", index=False)