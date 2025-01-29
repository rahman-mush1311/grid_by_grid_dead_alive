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
        key: log_values for key, values in dead_pdf_all_dict.items() 
        if (log_values := [np.log(value) for value in values if value > 0]) 
    
    }
    
    alive_log_pdf_dict ={
        key: log_values for key, values in alive_pdf_all_dict.items() 
        if (log_values := [np.log(value) for value in values if value > 0]) 
    
    }
    '''
    print("Log PDF  of Alive Dictionary:")
    for key, values in alive_log_pdf_dict.items():
        print(f"{key}: {values}")
    '''
    return dead_log_pdf_dict,alive_log_pdf_dict

        
def get_thresholds_from_roc(data):
    """
    this function selects thresholds from dead & alive log pdf's dictionary using roc curve, it filters furthers when the tpr improves fpr decreases, it rounds the fpr, tpr values to 2 digits get more relevant thresholds
    Parameters:
    - data: dictionary containing {object id: log_pdf_values:[log of pdf_values_for displacements],true_labels: 0/1}
    Returns:
    - filtered threshold: list of threshold values got from filtering process
    
    """
    true_labels=[]
    log_pdf_values=[]
    for obj_data in data.values():
        log_pdf_values.extend(obj_data["log_values"])
        true_labels.extend([0 if obj_data["true_labels"] == 'a' else 1] * len(obj_data["log_values"]))
        #print(log_pdf_values,true_labels)
        #print(len(log_pdf_values),len(true_labels))
    
    true_labels=np.array(true_labels)
    log_pdf_values=np.array(log_pdf_values)
    print(true_labels.shape,log_pdf_values.shape)
    fpr, tpr, roc_thresholds = roc_curve(true_labels, log_pdf_values)
    #print(min(roc_thresholds),max(roc_thresholds))
    
    filtered_thresholds = []
    for i in range(1, len(roc_thresholds)):
        if tpr[i] > tpr[i - 1] or fpr[i] < fpr[i - 1]:
            #print(tpr[i],tpr[i - 1], fpr[i] ,fpr[i - 1])
            filtered_thresholds.append(roc_thresholds[i])
    '''
    plt.figure(figsize=(10, 6))
    # Plot the ROC curve
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.show()
    '''
    #print(len(filtered_thresholds))
    #print(filtered_thresholds)
    return filtered_thresholds
def get_threshold_from_object_minimum(data):
    """
    this function selects thresholds from dead & alive log pdf's dictionary taking the minimum of each objects, it filters furthers when the tpr improves fpr decreases, it also produces an rox curve for the minimum's
    Parameters:
    -data: dictionary containing {object id: log_pdf_values:[log of pdf_values_for displacements],true_labels: 0/1}
    
    Returns:
    - filtered threshold: list of threshold values got from filtering process
    
    """
    thresholds=[ min(obj_data["log_values"]) for obj_id, obj_data in data.items() if obj_data["log_values"]]
    #print(thresholds)
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
        file.write(f"confusion matrices roc_curve for window size 1 & threshold {threshold}:\n")
        file.write(f"{cm[0][0]:} {cm[0][1]:}\n")
        file.write(f"{cm[1][0]:} {cm[1][1]:}\n")
        file.write("\n")
        
        accuracy = accuracy_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)

        file.write(f"Accuracy: , {accuracy} \n")
        file.write(f"F1-score:, {f1}\n")
        file.write(f"Recall:, {recall}\n")

def combine_dictionaries(dead_log_pdf_dict,alive_log_pdf_dict):
    """
    this function merges the dictionaries, it modifies the key with a/d indicating coming from dead or alive
    Parameters:
    - dead_log_pdf_dict: dictionary containing {object id: (log of pdf_values_for displacements) for all the dead displacements}
    - alive_log_pdf_dict: dictionary containing {object id: (log of pdf_values_for displacements) for all the alive displacements}
    Returns:
    - alive_log_pdf_dict: dictionary containing {object id: (log of pdf_values_for displacements) for all the displacements}
    
    """
    assert all(values and all(value < 0 for value in values) for values in alive_log_pdf_dict.values()),\
        "Error: Found empty lists or positive values in alive_log_pdf_dict"
    
    assert all(values and all(value < 0 for value in values) for values in dead_log_pdf_dict.values()),\
        "Error: Found empty lists or positive values in dead_log_pdf_dict"

    merged_dict={}
    for obj_id, log_pdfs in dead_log_pdf_dict.items():
        merged_dict[obj_id]={"log_values": log_pdfs, "true_labels":'d'
        }
    for obj_id, log_pdfs in alive_log_pdf_dict.items():
        merged_dict[obj_id]={"log_values": log_pdfs, "true_labels":'a'
        }
    
    #print(len(merged_dict))
    return merged_dict
    

def thresholding_with_window_roc_curve(data,window):
    """
    this function does the classification according to the thresholds got from roc_curve, it looks at different window to consider the classification 
    additionally it computes the thresholds for which we get the maximum tp/tn
    Parameters:
    - data: dictionary containing {object id: log_pdf_values:[log of pdf_values_for displacements],true_labels: 0/1}
    Returns:
    - N/A
    """
    #filtered_thresholds=get_thresholds_from_roc(data)
    filtered_thresholds=get_threshold_from_object_minimum(data)
    
    window_size=window
    thresholds=filtered_thresholds
    #print(thresholds)
    
    best_threshold = None
    best_accuracy_threshold= None
    best_classify = -float('inf')
    best_accuracy = -float('inf')
    
    for threshold in thresholds:
        predictions=[]
        true_labels=[]
        for obj_id in data:
            cls = 'd'
            for i in range(len(data[obj_id]["log_values"]) - window_size + 1):
                w = data[obj_id]["log_values"][i:i+window_size]
                #print(len(data[obj_id]["log_values"]))
                #print(data[obj_id]["log_values"])
                #print(i,w)
                if all([p <= threshold for p in w]):
                    cls = 'a'
            predictions.append(0 if cls=='a' else 1)
            true_labels.append (0 if data[obj_id]["true_labels"] == 'a' else 1)
        cm = confusion_matrix(true_labels, predictions, labels=[0, 1])
        
        #print(len(true_labels),len(predictions))
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        classify = cm[0, 0] + cm[1, 1]  # True positives + True negatives
        print(f"{threshold:<12.3f}{accuracy:<10.3f}{f1:<10.3f}{recall:<10.3f}{cm.tolist()} from ROC Curve {window_size}")
        
        if classify > best_classify:
            best_classify = classify
            best_threshold = threshold
        if best_accuracy < accuracy:
            best_accuracy_threshold = threshold
            best_accuracy=accuracy
            
    print(f"Optimal Threshold: {best_threshold}, Maximum Classification: {best_classify}, Accuracy: {best_accuracy}, Threshold for Best Accuracy: {best_accuracy_threshold}")
    
    
    return best_threshold
    
def predict_probabilities_dictionary_update(data, best_threshold,window_size):

    for obj_id in data:
        cls = 'd'
        for i in range(len(data[obj_id]["log_values"]) - window_size + 1):
            w = data[obj_id]["log_values"][i:i+window_size]
            if all([p <= best_threshold for p in w]):
                cls = 'a'

        # Update the dictionary with predicted and true labels
        data[obj_id] = {
            'log_pdfs': data[obj_id]["log_values"],  # Original log PDF values
            'true_labels': data[obj_id]["true_labels"],
            'predicted_labels': cls
        }
    print(data)
    return data

      
      
   