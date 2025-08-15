import pandas as pd
import numpy as np
import os
from sklearn.metrics import (
    precision_score, recall_score, 
    confusion_matrix, accuracy_score, f1_score, 
    roc_auc_score, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def plot_confusion_matrix(cm, title, model_name, save_path=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Stable (0)', 'Non-stable (1)'],
                yticklabels=['Stable (0)', 'Non-stable (1)'])
    plt.title(f'{title} - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved: {save_path}")
    
    plt.show()

def evaluate_model_performance(csv_file):
    """评估YOLO模型的性能"""
    print("="*80)
    print("MODEL PERFORMANCE EVALUATION")
    print("="*80)
    

    df = pd.read_csv(csv_file)
    print(f"Total samples loaded: {len(df)}")
    

    true_labels = (df['stability'] == 'non-stable').astype(int)
    
 
    vmsl_predictions = df['vmsl_prediction']
    vmsl_confidence = df['vmsl_confidence']
    
    print(f"True labels distribution:")
    print(f"  Stable (0): {(true_labels == 0).sum()}")
    print(f"  Non-stable (1): {(true_labels == 1).sum()}")
    
    print("\n" + "="*60)
    print("VMSL MODEL PERFORMANCE")
    print("="*60)
    
 
    vmsl_accuracy = accuracy_score(true_labels, vmsl_predictions)
    vmsl_precision = precision_score(true_labels, vmsl_predictions, average='binary')
    vmsl_recall = recall_score(true_labels, vmsl_predictions, average='binary')
    vmsl_specificity = calculate_specificity(true_labels, vmsl_predictions)
    vmsl_f1 = f1_score(true_labels, vmsl_predictions, average='binary')
    
 
    try:
        vmsl_probs = np.where(vmsl_predictions == 1, vmsl_confidence, 1 - vmsl_confidence)
        vmsl_auc = roc_auc_score(true_labels, vmsl_probs)
    except:
        vmsl_auc = 0.5
    
   
    vmsl_composite_score = vmsl_precision * vmsl_specificity * vmsl_recall
    

    vmsl_cm = confusion_matrix(true_labels, vmsl_predictions)
    
    print(f"Accuracy:     {vmsl_accuracy:.4f}")
    print(f"Precision:    {vmsl_precision:.4f}")
    print(f"Recall:       {vmsl_recall:.4f}")
    print(f"Specificity:  {vmsl_specificity:.4f}")
    print(f"F1-Score:     {vmsl_f1:.4f}")
    print(f"AUC:          {vmsl_auc:.4f}")
    print(f"P*S*R Score:  {vmsl_composite_score:.4f}")
    print(f"\nConfusion Matrix:")
    print(vmsl_cm)
    
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*60)
    
    print("\nVMSL Model:")
    print(classification_report(true_labels, vmsl_predictions, 
                              target_names=['Stable', 'Non-stable'], digits=4))
    

    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    

    plt.figure(figsize=(8, 6))
    sns.heatmap(vmsl_cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=['Stable (0)', 'Non-stable (1)'],
                yticklabels=['Stable (0)', 'Non-stable (1)'])
    plt.title('VMSL Model Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'vmsl_confusion_matrix_{timestamp}.png', dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved as: vmsl_confusion_matrix_{timestamp}.png")
    
    plt.show()
    

    results_summary = {
        'Model': ['VMSL'],
        'Accuracy': [vmsl_accuracy],
        'Precision': [vmsl_precision],
        'Recall': [vmsl_recall],
        'Specificity': [vmsl_specificity],
        'F1_Score': [vmsl_f1],
        'AUC': [vmsl_auc],
        'PSR_Score': [vmsl_composite_score]
    }
    
    results_df = pd.DataFrame(results_summary)
    results_filename = f'vmsl_model_evaluation_results_{timestamp}.csv'
    results_df.to_csv(results_filename, index=False)
    print(f"\nDetailed results saved to: {results_filename}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETED")
    print("="*80)
    
    return {
        'vmsl_metrics': {
            'accuracy': vmsl_accuracy,
            'precision': vmsl_precision,
            'recall': vmsl_recall,
            'specificity': vmsl_specificity,
            'f1': vmsl_f1,
            'auc': vmsl_auc,
            'psr_score': vmsl_composite_score,
            'confusion_matrix': vmsl_cm
        }
    }

def analyze_confidence_distribution(csv_file):

    print("\n" + "="*60)
    print("CONFIDENCE DISTRIBUTION ANALYSIS")
    print("="*60)
    
    df = pd.read_csv(csv_file)
    

    print("VMSL Model Confidence:")
    print(f"  Mean: {df['vmsl_confidence'].mean():.4f}")
    print(f"  Std:  {df['vmsl_confidence'].std():.4f}")
    print(f"  Min:  {df['vmsl_confidence'].min():.4f}")
    print(f"  Max:  {df['vmsl_confidence'].max():.4f}")
    
  
    plt.figure(figsize=(8, 6))
    plt.hist(df['vmsl_confidence'], bins=30, alpha=0.7, color='orange', edgecolor='black')
    plt.title('VMSL Model Confidence Distribution')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.axvline(df['vmsl_confidence'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["vmsl_confidence"].mean():.3f}')
    plt.legend()
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'vmsl_confidence_distribution_{timestamp}.png', dpi=300, bbox_inches='tight')
    print(f"Confidence distribution plot saved as: vmsl_confidence_distribution_{timestamp}.png")
    
    plt.show()

if __name__ == "__main__":

    csv_file = "predictions_results.csv"
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found!")
        print("Please make sure the predictions file exists.")
        exit(1)
    
    try:
        # 主要评估
        results = evaluate_model_performance(csv_file)
        
        # 置信度分析
        analyze_confidence_distribution(csv_file)
        
        print("\n🎉 Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()