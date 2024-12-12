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

from grid_by_grid_guassian_estimation import grid_by_grid_displacement_observation,grid_covariance_calculate,print_grid_stats
from result_visualization import mean_covariance_plot,make_collage
from evaluate_with_probability_density_values import grid_by_grid_pdf,calculate_pdf_all_by_displacements,get_pdf_value_list,mismatching_pdf_observations,get_unique_values_of_pdfs,alive_dead_thresholding,alive_dead_thresholding_sequential,evaluate_model_performance
from evaluate_thresholding_model import get_log_dictionary

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
    
def compute_likelihood(log_pdfs_dead_dis_dead, log_pdfs_dead_dis_alive, true_labels, prior_dead, prior_alive):
    """
    this function creates a log of all the pdfs values displacements from dead / alive  log pdf's dictionary does the classification which ever's sum is greater
    Parameters:
    - log_pdfs_dead_dis_dead: dictionary containing {object id: (pdf_values_for displacements using dead grid stats)}
    - log_pdfs_dead_dis_dead: dictionary containing {object id: (pdf_values_for displacements) using alive grid stats}
    Returns:
    - curr_likelihood: dictionary containing {object id: {true_label: 'd'/'a', predicted label: 'd'/'a'}}
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

        # Compute total for normalization
        total = (
            np.sum(valid_dead_log_pdfs) * prior_dead +
            np.sum(valid_alive_log_pdfs) * prior_alive
        )

        if total == 0 or np.isnan(total):
            print(f"Warning: Total is invalid for obj_id {obj_id}. Skipping...")
            continue

        # Compute the log posterior probabilities
        dead_log_sum_pdf = (np.sum(valid_dead_log_pdfs) * prior_dead) / total
        alive_log_sum_pdf = (np.sum(valid_alive_log_pdfs) * prior_alive) / total
        
        #print(f"dead_log_sum is {dead_log_sum_pdf}, alive_log_sum_pdf: {alive_log_sum_pdf}")
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
    
    print(curr_likelihood)
    return curr_likelihood


    
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
    print_grid_stats(dead_grid_stats)

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
    print(train_dead_with_dead_log_pdf)
    train_dead_with_alive_log_pdf,train_alive_with_alive_log_pdf=get_log_dictionary(train_dead_with_alive_pdf_dict,train_alive_with_alive_pdf_dict)
    
    prior_dead,prior_alive=calculate_prior(alive_train_obs,dead_train_obs)
    print(prior_dead,prior_alive)
    
    #classifications
    dead_obs_pred=compute_likelihood(train_dead_with_dead_log_pdf,train_dead_with_alive_log_pdf,'d',prior_dead,prior_alive)
    alive_obs_pred=compute_likelihood(train_alive_with_dead_log_pdf,train_alive_with_alive_log_pdf,'a',prior_dead,prior_alive)
    

