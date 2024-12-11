from scipy.stats import multivariate_normal
import numpy as np
import pandas as pd
import os
import logging
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay,accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def get_log_dictionary(dead_pdf_all_dict,alive_pdf_all_dict):
    
    dead_log_pdf_dict ={
        key: [np.log(value) for value in values if value > 0]
        for key,values in dead_pdf_all_dict.items()
    
    }
    
    alive_log_pdf_dict ={
        key: [np.log(value) for value in values if value > 0]
        for key,values in alive_pdf_all_dict.items()
    
    }
    '''
    print("Log PDF  of Dead Dictionary:")
    for key, values in alive_log_pdf_dict.items():
        print(f"{key}: {values}")
    '''
    return dead_log_pdf_dict,alive_log_pdf_dict

        
def get_thresholds_from_roc(dead_log_pdf_dict,alive_log_pdf_dict):

    alive_log_values = [v for values in alive_log_pdf_dict.values() for v in values]
    dead_log_values = [v for values in dead_log_pdf_dict.values() for v in values]
    
    true_labels = [0] * len(alive_log_values) + [1] * len(dead_log_values)  # 1 for Alive, 0 for Dead
    log_pdf_values = alive_log_values + dead_log_values
    
    fpr, tpr, roc_thresholds = roc_curve(true_labels, log_pdf_values)
    #print(roc_thresholds)
    filtered_thresholds = []
    for i in range(1, len(roc_thresholds)):
        if tpr[i] > tpr[i - 1] or fpr[i] < fpr[i - 1]:
            filtered_thresholds.append(roc_thresholds[i])
    print(len(filtered_thresholds))
    write_confusion_matrices('confusion_matrices_roc_curve.txt',filtered_thresholds,true_labels,log_pdf_values)
    
def write_confusion_matrices(filename,filtered_thresholds,true_labels,log_pdf_values):
    
    with open(filename, "w") as file:
        for threshold in filtered_thresholds:
            predicted_labels = [0 if log_pdf <= threshold else 1 for log_pdf in log_pdf_values]

            # Compute confusion matrix
            cm = confusion_matrix(true_labels, predicted_labels)

            # Write confusion matrix to the file
            file.write(f"Threshold: {threshold:}\n")
            file.write("Confusion Matrix:\n")
            file.write(f"{cm[0][0]:} {cm[0][1]:} (True Alive 0)\n")
            file.write(f"{cm[1][0]:} {cm[1][1]:} (True Dead 1)\n")
            file.write("\n")
            
def thresholding_with_window_size():
    print(f"working function")