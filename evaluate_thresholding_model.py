from scipy.stats import multivariate_normal
import numpy as np
import pandas as pd
import os
import logging
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay,accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

def get_log_dictionary(dead_pdf_all_dict,alive_pdf_all_dict):
    """
    this function creates a log of all the pdfs values displacements from dead & alive pdf's dictionary
    Parameters:
    - dead_pdf_all_dict: dictionary containing {object id: (pdf_values_for displacements) for all the dead displacements}
    - alive_pdf_all_dict: dictionary containing {object id: (pdf_values_for displacements) for all the alive displacements}
    Returns:
    - dead_log_pdf_dict: dictionary containing {object id: (log of pdf_values_for displacements) for all the dead displacements}
    - alive_log_pdf_dict: dictionary containing {object id: (log of pdf_values_for displacements) for all the alive displacements}
    """
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
    """
    this function selects thresholds from dead & alive log pdf's dictionary using roc curve, it filters furthers when the tpr improves fpr decreases, it rounds the fpr, tpr values to 2 digits get more relevant thresholds
    Parameters:
    - dead_log_pdf_dict: dictionary containing {object id: (log of pdf_values_for displacements) for all the dead displacements}
    - alive_log_pdf_dict: dictionary containing {object id: (log of pdf_values_for displacements) for all the alive displacements}
    Returns:
    - filtered threshold: list of threshold values got from filtering process
    
    """
    alive_log_values = [v for values in alive_log_pdf_dict.values() for v in values]
    dead_log_values = [v for values in dead_log_pdf_dict.values() for v in values]
    
    true_labels = [0] * len(alive_log_values) + [1] * len(dead_log_values)  # 0 for Alive, 1 for Dead
    log_pdf_values = alive_log_values + dead_log_values
    
    fpr, tpr, roc_thresholds = roc_curve(true_labels, log_pdf_values)
    
    fpr=[round(fp,2) for fp in fpr]
    tpr=[round(tp,2) for tp in tpr]
    
    filtered_thresholds = []
    for i in range(1, len(roc_thresholds)):
        if tpr[i] > tpr[i - 1] or fpr[i] < fpr[i - 1]:
            #print(tpr[i],tpr[i - 1], fpr[i] ,fpr[i - 1])
            filtered_thresholds.append(roc_thresholds[i])
    print(len(filtered_thresholds))
    return filtered_thresholds
def get_threshold_from_object_minimum(data,dead_log_pdf_dict,alive_log_pdf_dict):
    """
    this function selects thresholds from dead & alive log pdf's dictionary taking the minimum of each objects, it filters furthers when the tpr improves fpr decreases, it also produces an rox curve for the minimum's
    Parameters:
    -data: combined dictionary with all the alive & dead displacements
    - dead_log_pdf_dict: dictionary containing {object id: (log of pdf_values_for displacements) for all the dead displacements}
    - alive_log_pdf_dict: dictionary containing {object id: (log of pdf_values_for displacements) for all the alive displacements}
    Returns:
    - filtered threshold: list of threshold values got from filtering process
    
    """
    thresholds=[min(values) for values in data.values()  if values ]
    thresholds.insert(0,float('inf'))
    
    alive_log_values = [v for values in alive_log_pdf_dict.values() for v in values]
    dead_log_values = [v for values in dead_log_pdf_dict.values() for v in values]
    true_labels = [0] * len(alive_log_values) + [1] * len(dead_log_values)  # 1 for Alive, 0 for Dead
    log_pdf_values = alive_log_values + dead_log_values
    
    tprs, fprs = [], []
    for threshold in thresholds:
        pred = np.array(log_pdf_values) <= threshold
        pred = pred.astype(int)
        tn, fp, fn, tp = confusion_matrix(true_labels, pred).ravel()
        tprs.append(tp / (tp + fn))
        fprs.append(fp / (fp + tn))
    
    sorted_indices = sorted(range(len(thresholds)), key=lambda i: thresholds[i], reverse=True)
    thresholds = [thresholds[i] for i in sorted_indices]
    tprs = [tprs[i] for i in sorted_indices]
    fprs = [fprs[i] for i in sorted_indices]
    
    plt.figure(figsize=(10, 6))
    # Plot the ROC curve
    plt.plot(fprs, tprs)
    
    # Truncate thresholds to two digits and format labels
    truncated_thresholds = [f"{round(thresh, 2)}" for thresh in thresholds]
    plt.scatter(fprs, tprs, c='red', label='Thresholds', s=20)
    for i, thresh in enumerate(truncated_thresholds):
        plt.annotate(thresh, (fprs[i], tprs[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    #plt.show()
    #print(len(thresholds))
    
    return thresholds
def write_confusion_matrices(filename,threshold,true_labels,predicted_labels):
    """
    this function writes the confusion matrices to the files according to thresholds
    Parameters:
    - filename: string of the filenames
    - threshold: current threshold to get the confusion matrix
    - true_labels: 0/1 indicating dead/alive
    - predicted_labels: 0/1 calculated using the threshold
    Returns:
    - N/A
    
    """
    with open(filename, "a") as file:
        #for threshold in filtered_thresholds:
            #predicted_labels = [0 if log_pdf <= threshold else 1 for log_pdf in log_pdf_values]

            # Compute confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        # Write confusion matrix to the file
        file.write(f"confusion matrices from minimum object level for window size 2 & threshold {threshold}:\n")
        file.write(f"{cm[0][0]:} {cm[0][1]:}\n")
        file.write(f"{cm[1][0]:} {cm[1][1]:}\n")
        file.write("\n")

def combine_dictionaries(dead_log_pdf_dict,alive_log_pdf_dict):
    """
    this function merges the dictionaries, it modifies the key with a/d indicating coming from dead or alive
    Parameters:
    - dead_log_pdf_dict: dictionary containing {object id: (log of pdf_values_for displacements) for all the dead displacements}
    - alive_log_pdf_dict: dictionary containing {object id: (log of pdf_values_for displacements) for all the alive displacements}
    Returns:
    - alive_log_pdf_dict: dictionary containing {object id: (log of pdf_values_for displacements) for all the displacements}
    
    """
    merged_dict={}
    for obj_id, log_pdfs in dead_log_pdf_dict.items():
        merged_dict[f"{obj_id}d"]=log_pdfs
    for obj_id, log_pdfs in alive_log_pdf_dict.items():
        merged_dict[f"{obj_id}a"]=log_pdfs
    
    #print(len(merged_dict))
    return merged_dict
    
def max_matrix(threshold,window_size,true_labels,predicted_labels):
    """
    this function prints the confusion matrices which has highest TP/TN rate
    Parameters:
    - threshold: current threshold to get the confusion matrix
    - window_size: for which size we are getting the maximum TP/TN rate
    - true_labels: 0/1 indicating dead/alive
    - predicted_labels: 0/1 calculated using the threshold
    Returns:
    - N/A
    
    """
    cm = confusion_matrix(true_labels, predicted_labels)
    print(f"Confusion Matrix for Threshold {threshold} Maximum TP/TN & Window Size {window_size}:")
    print(f"True d Pred d [[{cm[0][0]} {cm[0][1]}] True d Pre a")  # True Positive, False Negative
    print(f"True a Pred d [{cm[1][0]} {cm[1][1]}]] True a Pre a")  # False Positive, True Negative

def thresholding_with_window_roc_curve(dead_log_pdf_dict,alive_log_pdf_dict):
    """
    this function does the classification according to the thresholds got from roc_curve, it looks at different window to consider the classification 
    additionally it computes the thresholds for which we get the maximum tp/tn
    Parameters:
    - dead_log_pdf_dict: dictionary containing {object id: (log of pdf_values_for displacements) for all the dead displacements}
    - alive_log_pdf_dict: dictionary containing {object id: (log of pdf_values_for displacements) for all the alive displacements}
    Returns:
    - N/A
    """
    data=combine_dictionaries(dead_log_pdf_dict,alive_log_pdf_dict)
    filtered_thresholds=get_thresholds_from_roc(dead_log_pdf_dict,alive_log_pdf_dict)
    #get_threshold_from_object_minimum(data,dead_log_pdf_dict,alive_log_pdf_dict)
    
    window_size=2
    thresholds=filtered_thresholds[0:100]
    #print(thresholds)
    
    max_tp=0
    max_tn=0
    max_tp_threshold=0
    max_tn_threshold=0
    max_tp_pred=[]
    max_tp_true=[]
    max_tn_pred=[]
    max_tn_true=[]
    
    for threshold in thresholds:
        predictions=[]
        true_labels=[]
        #print(f"Threshold: {threshold} with window_size {window_size}:")
        for obj_id in data:
            cls = 'd'
            for i in range(len(data[obj_id]) - window_size + 1):
                w = data[obj_id][i:i+window_size]
                if all([p <= threshold for p in w]):
                    cls = 'a'
            #print(obj_id, cls, obj_id[-1])
            predictions.append(1 if cls=='a' else 0)
            true_labels.append (1 if obj_id[-1] == 'a' else 0)
        cm = confusion_matrix(true_labels, predictions, labels=[0, 1])
        '''
        #print(len(true_labels),len(predictions))
        # Print confusion matrix
        print("Confusion Matrix:")
        print(f"True d Pred d [[{cm[0][0]} {cm[0][1]}] True d Pre a")  # True Positive, False Negative
        print(f"True a Pred d [{cm[1][0]} {cm[1][1]}]] True a Pre a")  # False Positive, True Negative
        '''
        if(max_tp<cm[0][0]):
            max_tp=cm[0][0]
            max_tp_threshold=threshold
            max_tp_pred=predictions
            max_tp_true=true_labels
        if(max_tn<cm[1][1]):
            max_tn=cm[1][1]
            max_tn_threshold=threshold
            max_tn_pred=predictions
            max_tn_true=true_labels
        #print(max_tp,max_tn,max_tp_threshold,max_tn_threshold)
        #write_confusion_matrices('confusion_matrices_obj_level.txt',threshold,true_labels,predictions)
    print(max_tp,max_tn,max_tp_threshold,max_tn_threshold)
    max_matrix(max_tp_threshold,1,max_tp_true,max_tp_pred)
    max_matrix(max_tn_threshold,1,max_tn_true,max_tn_pred)

def thresholding_classification_with_window_minimum(dead_log_pdf_dict,alive_log_pdf_dict):
    """
    this function does the classification according to the thresholds got from taking object's minimum, it looks at different window to consider the classification 
    additionally it prints the confusion matrix
    Parameters:
    - dead_log_pdf_dict: dictionary containing {object id: (log of pdf_values_for displacements) for all the dead displacements}
    - alive_log_pdf_dict: dictionary containing {object id: (log of pdf_values_for displacements) for all the alive displacements}
    Returns:
    - N/A
    """
    data=combine_dictionaries(dead_log_pdf_dict,alive_log_pdf_dict)
    filtered_thresholds=get_threshold_from_object_minimum(data,dead_log_pdf_dict,alive_log_pdf_dict)
    
    window_size=2
    thresholds=filtered_thresholds[0:10]
    
    
    for threshold in filtered_thresholds:
        predictions=[]
        true_labels=[]
        print(f"Threshold: {threshold} with window_size {window_size}:")
        for obj_id in data:
            cls = 'd'
            for i in range(len(data[obj_id]) - window_size + 1):
                w = data[obj_id][i:i+window_size]
                if all([p <= threshold for p in w]):
                    cls = 'a'
            #print(obj_id, cls, obj_id[-1])
            predictions.append(1 if cls=='a' else 0)
            true_labels.append (1 if obj_id[-1] == 'a' else 0)
        cm = confusion_matrix(true_labels, predictions, labels=[0, 1])
        
        #print(len(true_labels),len(predictions))
        print("Confusion Matrix:")
        print(f"True d Pred d [[{cm[0][0]} {cm[0][1]}] True d Pre a")  # True Positive, False Negative
        print(f"True a Pred d [{cm[1][0]} {cm[1][1]}]] True a Pre a")  # False Positive, True Negative
        #write_confusion_matrices('confusion_matrices_obj_level.txt',threshold,true_labels,predictions)
      
      
   