import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
import matplotlib as mpl
from scipy.stats import skew
from PIL import Image
import os
import math
import random
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay,accuracy_score, precision_score, recall_score, f1_score

from grid_by_grid_guassian_estimation import grid_by_grid_displacement_observation,grid_covariance_calculate,print_grid_stats
from result_visualization import mean_covariance_plot,make_collage
from evaluate_with_probability_density_values import grid_by_grid_pdf,calculate_pdf_all_by_displacements,get_pdf_value_list,mismatching_pdf_observations,get_unique_values_of_pdfs
from evaluate_thresholding_model import get_log_dictionary,get_thresholds_from_roc,thresholding_with_window_roc_curve,thresholding_classification_with_window_minimum,predict_probabilities_dictionary_update

def prepare_train_test(curr_obs,train_ratio):
    """
    Splits a dictionary into train and test sets based on a specified ratio.
    
    Parameters:
    -curr_obs (dict): The input dictionary with keys as object IDs and values as observations (e.g., lists of log PDFs).
    -train_ratio (float): The ratio of the data to include in the training set (e.g., 0.8 for 80% train and 20% test).
    
    Returns:
    - train_dict: The training set dictionary.
    - test_dict: The test set dictionary.
    """
    TRAIN_RATIO=train_ratio
    keys = list(curr_obs.keys())
    random.shuffle(keys)

    # Calculate split index
    split_index = int(len(keys) * train_ratio)

    # Split keys and sort them
    train_keys = sorted(keys[:split_index])
    test_keys = sorted(keys[split_index:])

    # Create sorted train and test dictionaries
    train_dict = {key: curr_obs[key] for key in train_keys}
    test_dict = {key: curr_obs[key] for key in test_keys}

    return train_dict,test_dict

def calculate_prior(alive_train_obs,dead_train_obs):
    """
   calculate the prior probabilities according to the number of points, the forumla is: #of alive or dead objects/total number of object's (according to trainning)
    
    Parameters:
    -alive_train_obs: The dictionary with alive keys as object IDs and values as observations (e.g., lists of log PDFs).
    -dead_train_obs: The dictionary with dead keys as object IDs and values as observations (e.g., lists of log PDFs).
    
    Returns:
    - dead: the prior probability of dead
    - alive: the prior probability of alive
    """
    dead=len(dead_train_obs)/(len(dead_train_obs)+len(alive_train_obs))
    alive=len(alive_train_obs)/(len(dead_train_obs)+len(alive_train_obs))
    
    return dead,alive
    
def compute_likelihood_without_threshold(log_pdfs_dead_dis_dead, log_pdfs_dead_dis_alive, true_labels, prior_dead, prior_alive):
    """
    this function creates a log of all the pdfs values displacements from dead / alive  log pdf's dictionary and calculates the sum and adds the log of prior probabilites
    Parameters:
    - log_pdfs_dead_dis_dead: dictionary containing {object id: (pdf_values_for displacements using dead grid stats)}
    - log_pdfs_dead_dis_dead: dictionary containing {object id: (pdf_values_for displacements) using alive grid stats}
    Returns:
    - curr_likelihood: dictionary containing {object id: {true_label: 'd'/'a', dead_log_pdf: 'P(D/X), alive_log_pdf: P(A/X)'}}
    """
    curr_likelihood = {}
    
    for obj_id in log_pdfs_dead_dis_dead:
        cls = 'd'
        
        # Filter valid log PDFs
        valid_dead_log_pdfs = [v for v in log_pdfs_dead_dis_dead[obj_id] if v != 0]
        valid_alive_log_pdfs = [v for v in log_pdfs_dead_dis_alive[obj_id] if v != 0]
        
        #print(f"log_pdfs for obj_id {obj_id}. {valid_dead_log_pdfs} {valid_alive_log_pdfs}")
        
        if not valid_dead_log_pdfs or not valid_alive_log_pdfs:
            print(f"Warning: Invalid log_pdfs for obj_id {obj_id}. {valid_dead_log_pdfs} {valid_alive_log_pdfs}")
            continue
       
        # Compute the log posterior probabilities
        dead_log_sum_pdf = np.sum(valid_dead_log_pdfs) + np.log(prior_dead)
        alive_log_sum_pdf = np.sum(valid_alive_log_pdfs) + np.log(prior_alive)
        
        #print(f"dead_log_sum is {dead_log_sum_pdf}, alive_log_sum_pdf: {alive_log_sum_pdf}")
        '''
        # Classification based on posterior probabilities
        if dead_log_sum_pdf > alive_log_sum_pdf:
            cls = 'd'
            
        else:
            cls = 'a'
        
        # Record the true and predicted labels
        if true_labels == 'd':
            curr_likelihood[f"{obj_id}d"] = {'true_labels': 'd', 'predicted_labels': cls}
        else:
            curr_likelihood[f"{obj_id}a"] = {'true_labels': 'a', 'predicted_labels': cls}
        
        if true_labels == 'd':
            curr_likelihood[f"{obj_id}d"] = {'true_labels': 'd', 'dead_prob': dead_log_sum_pdf, 'alive_prob': alive_log_sum_pdf, 'predicted_labels': cls }
        else:
            curr_likelihood[f"{obj_id}a"] = {'true_labels': 'a', 'dead_prob': dead_log_sum_pdf, 'alive_prob': alive_log_sum_pdf, 'predicted_labels': cls }
        
        curr_likelihood[f"{obj_id}{true_labels}"] = {
            'true_labels': true_labels,
            'dead_prob': dead_log_sum_pdf,
            'alive_prob': alive_log_sum_pdf,
            'predicted_labels': cls
        }
        '''
        curr_likelihood[f"{obj_id}{true_labels}"] = {
            'true_labels': true_labels,
            'dead_prob': dead_log_sum_pdf,
            'alive_prob': alive_log_sum_pdf,
        }

    #print(curr_likelihood)
    
    return curr_likelihood



def create_confusion_matrix(data,t, thresholding_type,data_type):
    """
    Create a confusion matrix from two dictionaries containing alive and dead information.

    Parameters:
        alive_dict (dict): Dictionary with alive-related data (e.g., {obj_id: {'true_label': 'a', 'predicted_labels': 'd'}}).
        dead_dict (dict): Dictionary with dead-related data (e.g., {obj_id: {'true_label': 'd', 'predicted_labels': 'a'}}).

    Returns:
        None: Displays the confusion matrix.
    """
    '''
    # Combine dictionaries
    combined_dict = {**alive_dict, **dead_dict}

    # Extract true and predicted labels
    true_labels = [1 if v['true_labels'] == 'a' else 0 for v in combined_dict.values()]
    predicted_labels = [1 if v['predicted_labels'] == 'a' else 0 for v in combined_dict.values()]

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Dead (0)", "Alive (1)"])
    disp.plot(cmap="Blues")
    disp.ax_.set_title("Confusion Matrix")
    disp.ax_.set_xlabel("Predicted Labels")
    disp.ax_.set_ylabel("True Labels")
    #plt.show()

    # Print confusion matrix values
    print("Confusion Matrix:")
    print(cm)
    '''
    true_labels = [data[obj_id]['true_labels'] for obj_id in data]
    predicted_labels = [data[obj_id]['predicted_labels'] for obj_id in data]

    # Create the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=['d', 'a'])
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, pos_label='d', average='binary')
    recall = recall_score(true_labels, predicted_labels, pos_label='d', average='binary')
    
    print(f"{accuracy:<10.3f}{f1:<10.3f}{recall:<10.3f}")
    
    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Dead (0)", "Alive (1)"])
    disp.plot(cmap="Blues")
    disp.ax_.set_title(f" {data_type} Confusion Matrix Using {thresholding_type}")
    disp.ax_.set_xlabel("Predicted Labels")
    disp.ax_.set_ylabel("True Labels")
    
    metrics_text = (
        f"Accuracy: {accuracy:.3f}\n"
        f"F1-Score: {f1:.3f}\n"
        f"Recall: {recall:.3f}\n"
        f"Threshold: {t:.3f}"
    )
    disp.ax_.legend(
        handles=[
            plt.Line2D([], [], color='white', label=metrics_text)
        ],
        loc='lower right',
        fontsize=10,
        frameon=False
    )
    
    plt.show()
    
def plot_probabilities(data):
    '''
    this function just plots the P(D/X) vs P(A/X) for visualization purpose only
    
    Parameters:
    - data: dictionary containing object id: {{true_lables: 'a/d'},{dead_log_pdf: 'P(D/X)'},{alive_log_pdf: 'P(A/X)'}}
    
    Returns:
    - N/A
    '''
    x =[]
    y =[]
    colors = []
    
    for obj_id, values in data.items():
        x.append(values["dead_prob"])
        y.append(values["alive_prob"])
        colors.append("green" if values["true_labels"] == "a" else "red")

    # Create the plot
    plt.scatter(x, y, c=colors)
    plt.xlabel("P(D|X)")
    plt.ylabel("P(A|X)")
    plt.title("Probability Distribution by True Label")
    labels = ['True Label: a', 'True Label: d']

    # Add legend using proxy artists
    plt.legend(['green = alive, red = dead'])
    plt.show()


def optimize_threshold(data):
    """
    Finds the threshold that maximizes correct classification numbers. It takes the range between maximum & minimum difference of the probability values
    
    Parameters:
    - data: dictionary containing object id: {{true_lables: 'a/d'},{dead_log_pdf: 'P(D/X)'},{alive_log_pdf: 'P(A/X)'}}
    
    Returns:
    - best_threshold: for which threshold the accuracy/ classification rate is the largest
    """
    differences = [
        values['dead_prob'] - values['alive_prob']
        for values in data.values()
    ]
    min_diff = min(differences)
    max_diff = max(differences)
    threshold_dict={}
    # Define threshold range based on min and max differences
    thresholds = np.linspace(min_diff, max_diff, 100) 
    
    best_threshold = None
    best_classify = -float('inf')
    best_accuracy= -float('inf')
    best_accuracy_threshold = None 
    minimum_error=float('inf')
    for t in thresholds:
        predictions = {
            obj_id: 'd' if (values['dead_prob'] - values['alive_prob']) > t else 'a'
            for obj_id, values in data.items()
        }
        true_labels = [values['true_labels'] for values in data.values()]
        pred_labels = list(predictions.values())

        
        # Calculate confusion matrix and metrics
        cm = confusion_matrix(true_labels, pred_labels, labels=['d', 'a'])
        accuracy = accuracy_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels, pos_label='d', average='binary')
        recall = recall_score(true_labels, pred_labels, pos_label='d', average='binary')
        classify = cm[0, 0] + cm[1, 1]  # True positives + True negatives
        error=cm[0,1]+cm[1,0] #false positive + false negatives
      
        print(f"{t:<12.3f}{accuracy:<10.3f}{f1:<10.3f}{recall:<10.3f}{classify:<10}{cm.tolist()}")
        
        if classify > best_classify:
            best_classify = classify
            best_threshold = t
            minimum_error = error
            
        if best_accuracy < accuracy:
            best_accuracy_threshold = t
            best_accuracy=accuracy

    print(f"Optimal Threshold: {best_threshold}, Maximum Classification: {best_classify}, Accuracy: {best_accuracy}, Threshold for Best Accuracy: {best_accuracy_threshold}")
   
    return best_threshold

def compute_likelihood_with_threshold(data, threshold):
    """
    Classifies objects using precomputed probabilities and a given threshold.

    Parameters:
    - data: Dictionary with keys as object IDs and values as dictionaries containing:
        'dead_prob': Log-probability for dead.
        'alive_prob': Log-probability for alive.
        'true_labels': True label ('d' or 'a').
    - threshold: Decision boundary threshold.

    Returns:
    - curr_likelihood: Dictionary with object IDs and their true labels, predicted labels, and probabilities.
    """
    curr_likelihood = {}

    for obj_id, values in data.items():
        dead_prob = values['dead_prob']
        alive_prob = values['alive_prob']
        true_label = values['true_labels']

        # Classification using the threshold
        predicted_label = 'd' if (dead_prob - alive_prob) > threshold else 'a'

        # Store results
        curr_likelihood[obj_id] = {
            'true_labels': true_label,
            'dead_prob': dead_prob,
            'alive_prob': alive_prob,
            'predicted_labels': predicted_label
        }

    return curr_likelihood

def get_combined_dictionaries(dead_obs,alive_obs):
    '''
    this function takes 2 dictionaries & merges them into one by modifying the object id with a/d
    Parameters:
    - dead obs: dictionary containing dead observations: object id: [(frame,x,y)]
    - alive obs: dictionary containing alive observations: object id: [(frame,x,y)]
    Returns:
    - merged dict: object id: [(frame,x,y)]
    '''
    merged_dict={}
    
    for obj_id, paths in dead_obs.items():
        merged_dict[f"{obj_id}d"]=paths
    for obj_id, paths in alive_obs.items():
        merged_dict[f"{obj_id}a"]=paths
    
    #print(merged_dict)
    return merged_dict
def find_the_misclassified_obj(curr_pred_dict, curr_obs):
    '''
    this functions finds the misclassified object's ids only
    
    Parameters:
    - curr_pred_dict: object_id: {{true_labels: a/d}, {dead_log_prob: P(D/X)}, {alive_log_prob: P(A/X)}, {predicted_labels: a/d}}
    - curr_obs: object_id: [(frame,x,y)]
    
    Returns:
    - misclassified_ids: list containing the misclassified ids
    '''
    misclassified_ids = [ obj_id for obj_id, details in curr_pred_dict.items()
    if details["true_labels"] != details["predicted_labels"]
    ]
    print(misclassified_ids)
    return misclassified_ids
    
def plot_misclassified_paths(curr_pred_dict, curr_obs, misclassified_ids):
    '''
    the function plots the misclassified object's path
    
    Parameters:
    - curr_pred_dict: object_id: {{true_labels: a/d}, {dead_log_prob: P(D/X)}, {alive_log_prob: P(A/X)}, {predicted_labels: a/d}}
    - curr_obs: object_id: [(frame,x,y)]
    - misclassified_ids: list containing the misclassified ids
    
    Returns:
    N/A
    
    '''
    for obj_id in misclassified_ids:
        #obj_id = misclassified_ids[0]  # Take the first misclassified object
        path = curr_obs[obj_id]
        if len(path)>2:
            frames, x, y = zip(*path)
    
            plt.figure(figsize=(10, 6))
            plt.plot(x, y, marker='o', label=f'{obj_id} (True: {curr_pred_dict[obj_id]["true_labels"]}, Pred: {curr_pred_dict[obj_id]["predicted_labels"]})')
    
            # Enhance the plot
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Path of a Misclassified Object')
            plt.legend()
            plt.grid(True)
            plt.show()

def prepare_data(dead_obs,alive_obs):
    """
    this functions calls all the related functions for calculating the bayesian probability
    - dead_obs : dictionary of dead coordinates {object id: [frame,x,y]}
    - alive_obs : dictionary of alive coordinates {object id: [frame,x,y]}
    """
    #splits into train test for alive & dead
    alive_train_obs,alive_test_obs=prepare_train_test(alive_obs,0.8)
    dead_train_obs,dead_test_obs=prepare_train_test(dead_obs,0.8)
    #print(len(alive_train_obs),len(alive_test_obs))
    #print(len(dead_train_obs),len(dead_test_obs))
    
    #grid_by grid displacements & mu_sigma calculation for dead
    dead_train_grid_displacements=grid_by_grid_displacement_observation(dead_train_obs,5,4128,2196)   
    dead_grid_stats=grid_covariance_calculate(dead_train_grid_displacements)
    #print_grid_stats(dead_grid_stats)

    #grid_by grid displacements & mu_sigma calculation for alive
    alive_train_grid_displacements=grid_by_grid_displacement_observation(alive_train_obs,5,4128,2196)   
    alive_grid_stats=grid_covariance_calculate(alive_train_grid_displacements)
    print_grid_stats(alive_grid_stats)

    #grid by grid pdf calculation with dead & alive grid stats 
    train_dead_with_dead_pdf_dict=calculate_pdf_all_by_displacements(dead_train_obs,dead_grid_stats,4128,2196)
    print(len(train_dead_with_dead_pdf_dict))
    train_dead_with_alive_pdf_dict=calculate_pdf_all_by_displacements(dead_train_obs,alive_grid_stats,4128,2196)
    print(len(train_dead_with_alive_pdf_dict))
    train_alive_with_dead_pdf_dict=calculate_pdf_all_by_displacements(alive_train_obs,dead_grid_stats,4128,2196)
    print(len(train_alive_with_dead_pdf_dict))
    train_alive_with_alive_pdf_dict=calculate_pdf_all_by_displacements(alive_train_obs,alive_grid_stats,4128,2196)
    print(len(train_alive_with_alive_pdf_dict))
    
    #log of the grid by grid pdfs
    train_dead_with_dead_log_pdf,train_alive_with_dead_log_pdf=get_log_dictionary(train_dead_with_dead_pdf_dict,train_alive_with_dead_pdf_dict)
    #print(train_dead_with_dead_log_pdf)
    train_dead_with_alive_log_pdf,train_alive_with_alive_log_pdf=get_log_dictionary(train_dead_with_alive_pdf_dict,train_alive_with_alive_pdf_dict)
    
    prior_dead,prior_alive=calculate_prior(alive_train_obs,dead_train_obs)
    print(prior_dead,prior_alive)
    
    #calculates the sum of log probabilites with prior
    dead_obs_pred=compute_likelihood_without_threshold(train_dead_with_dead_pdf_dict,train_dead_with_alive_pdf_dict,'d',prior_dead,prior_alive)
    alive_obs_pred=compute_likelihood_without_threshold(train_alive_with_dead_pdf_dict,train_alive_with_alive_pdf_dict,'a',prior_dead,prior_alive)
    #concates both of the precomputed dictionaries
    total_preds=dead_obs_pred | alive_obs_pred

   
    #thresholding related classification with scores
    #thresholding_with_window_roc_curve(train_dead_with_dead_log_pdf,train_alive_with_dead_log_pdf)
    
    #finds the threshold by maximizing accuracy
    threshold=optimize_threshold(total_preds)
    total_preds_thresholds=compute_likelihood_with_threshold(total_preds, threshold) #calculates the classifications
    #create_confusion_matrix(total_preds_thresholds,threshold)
    
    #test set probability calculations
    test_dead_with_dead_pdf_dict=calculate_pdf_all_by_displacements(dead_test_obs,dead_grid_stats,4128,2196)
    test_dead_with_alive_pdf_dict=calculate_pdf_all_by_displacements(dead_test_obs,alive_grid_stats,4128,2196)
    
    test_alive_with_dead_pdf_dict=calculate_pdf_all_by_displacements(alive_test_obs,dead_grid_stats,4128,2196)
    test_alive_with_alive_pdf_dict=calculate_pdf_all_by_displacements(alive_test_obs,alive_grid_stats,4128,2196)
    
    #log of the grid by grid pdfs
    test_dead_with_dead_log_pdf,test_alive_with_dead_log_pdf=get_log_dictionary(test_dead_with_dead_pdf_dict,test_alive_with_dead_pdf_dict)
    #print(train_dead_with_dead_log_pdf)
    test_dead_with_alive_log_pdf,test_alive_with_alive_log_pdf=get_log_dictionary(test_dead_with_alive_pdf_dict,test_alive_with_alive_pdf_dict)
    
    #calculates the log sum of the probabilites with prior
    test_dead_obs_pred=compute_likelihood_without_threshold(test_dead_with_dead_pdf_dict,test_dead_with_alive_pdf_dict,'d',prior_dead,prior_alive)
    test_alive_obs_pred=compute_likelihood_without_threshold(test_alive_with_dead_pdf_dict,test_alive_with_alive_pdf_dict,'a',prior_dead,prior_alive)
    
    test_total=test_dead_obs_pred | test_alive_obs_pred
    test_total_preds_thresholds=compute_likelihood_with_threshold(test_total, threshold)#does classification based on threshold
    #create_confusion_matrix(test_total_preds_thresholds,threshold, 'Bayesian', 'Testing Data')
    
    #thresholding related classification with scores
    '''
    roc_threshold_1=thresholding_with_window_roc_curve(train_dead_with_dead_log_pdf,train_alive_with_dead_log_pdf,2)
    roc_train_window_1=predict_probabilities_dictionary_update(train_dead_with_dead_log_pdf,train_alive_with_dead_log_pdf, roc_threshold_1,2)
    create_confusion_matrix(roc_train_window_1,roc_threshold_1,'ROC Curve', 'Trainning Data')
    roc_test_window_1=predict_probabilities_dictionary_update(test_dead_with_dead_log_pdf,test_alive_with_dead_log_pdf, roc_threshold_1,2)
    create_confusion_matrix(roc_test_window_1,roc_threshold_1, 'ROC Curve', 'Testing Data')
    
    obj_minimum_threshold_1=thresholding_classification_with_window_minimum(train_dead_with_dead_log_pdf,train_alive_with_dead_log_pdf,1)
    obj_train_window_1=predict_probabilities_dictionary_update(train_dead_with_dead_log_pdf,train_alive_with_dead_log_pdf,obj_minimum_threshold_1,1)
    create_confusion_matrix(obj_train_window_1,obj_minimum_threshold_1, 'Object Minimum', 'Trainning Data')
    obj_test_window_1=predict_probabilities_dictionary_update(test_dead_with_dead_log_pdf,test_alive_with_dead_log_pdf,obj_minimum_threshold_1,1)
    create_confusion_matrix(obj_test_window_1,obj_minimum_threshold_1, 'Object Minimum', 'Testing Data')
    '''
    
    train_obs=get_combined_dictionaries(dead_train_obs,alive_train_obs)
    mis_bayes=find_the_misclassified_obj(total_preds_thresholds, train_obs)
    roc_train_window_1=predict_probabilities_dictionary_update(train_dead_with_dead_log_pdf,train_alive_with_dead_log_pdf, -9.971,1)
    miss_roc_1=find_the_misclassified_obj(roc_train_window_1, train_obs)
    roc_train_window_2=predict_probabilities_dictionary_update(train_dead_with_dead_log_pdf,train_alive_with_dead_log_pdf, -7.616,2)
    miss_roc_2=find_the_misclassified_obj(roc_train_window_2, train_obs)
    obj_train_window_1=predict_probabilities_dictionary_update(train_dead_with_dead_log_pdf,train_alive_with_dead_log_pdf,-9.052,1)
    miss_obj_1=find_the_misclassified_obj(obj_train_window_1, train_obs)
    obj_train_window_2=predict_probabilities_dictionary_update(train_dead_with_dead_log_pdf,train_alive_with_dead_log_pdf,-7.602,2)
    miss_obj_2=find_the_misclassified_obj(obj_train_window_2, train_obs)
    
    #plot_misclassified_paths(total_preds_thresholds, train_obs, mis_bayes,miss_roc_1,miss_roc_2,miss_obj_1,miss_obj_2)
    
    #plot_misclassified_paths(total_preds_thresholds, train_obs, mis_bayes)
    #plot_misclassified_paths(roc_train_window_1, train_obs, miss_roc_1)
    
    test_obs=get_combined_dictionaries(dead_test_obs,alive_test_obs)
    #plot_misclassified_paths(test_total_preds_thresholds, test_obs)
    '''
    roc_train_window_1=predict_probabilities_dictionary_update(train_dead_with_dead_log_pdf,train_alive_with_dead_log_pdf, -9.971,1)
    plot_misclassified_paths(roc_train_window_1, train_obs)
    roc_test_window_1=predict_probabilities_dictionary_update(test_dead_with_dead_log_pdf,test_alive_with_dead_log_pdf,  -9.971,1)
    plot_misclassified_paths(roc_test_window_1, test_obs)
    
    roc_train_window_2=predict_probabilities_dictionary_update(train_dead_with_dead_log_pdf,train_alive_with_dead_log_pdf, -7.616,2)
    plot_misclassified_paths(roc_train_window_2, train_obs)
    roc_test_window_2=predict_probabilities_dictionary_update(test_dead_with_dead_log_pdf,test_alive_with_dead_log_pdf,  -7.616,2)
    plot_misclassified_paths(roc_test_window_2, test_obs)
    '''