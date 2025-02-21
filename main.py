# Import packages after ensuring they're installed

import sys
import importlib
import random
import re
import collections
import math
from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import custom functions
from grid_by_grid_guassian_estimation import load_observations,grid_by_grid_displacement_observation,grid_covariance_calculate,print_grid_stats,grid_by_grid_observation,convert_dict_to_dataframe,combine_df_write_to_csv,prepare_data
from evaluate_with_probability_density_values import grid_by_grid_pdf,calculate_pdf_all_by_displacements,get_pdf_value_list,mismatching_pdf_observations,get_unique_values_of_pdfs
from result_visualization import plot_pdf_histogram_bins,plot_pdf_overlay_histogram_bins,mean_covariance_plot,plot_cdf_line_side_by_side,plot_cdf_line_with_log_side_by_side,plot_cdf_line_overlay,plot_cdf_min_max_normalized_line_overlay,plot_cdf_zscore_normalized_line_overlay,filtered_overlay_histogram,large_small_frequency_overlay_histogram,make_collage
from evaluate_thresholding_model import get_log_dictionary,get_thresholds_from_roc,thresholding_with_window_roc_curve,get_threshold_from_object_minimum,combine_dictionaries,predict_probabilities_dictionary_update
from evaluate_with_bayesian import prepare_data,create_confusion_matrix
# Main code execution
if __name__ == "__main__":
    
    #following 4 functions calculates grid by grid mu & covariance matrix
    dead_file_list = ['DeadObjectXYs.txt']
    dead_obs = load_observations(dead_file_list) 
    #print(dead_obs)
    dead_grid_displacements=grid_by_grid_displacement_observation(dead_obs,5,4128,2196)   
    dead_grid_stats=grid_covariance_calculate(dead_grid_displacements,"dead")
    #print_grid_stats(dead_grid_stats) #run this to see the mu & sigma in formatted way
    
    #in below statement calculates the all dead object's probability density value of the displacements
    dead_pdf_all_dict=calculate_pdf_all_by_displacements(dead_obs,dead_grid_stats,4128,2196)
    #print(dead_pdf_all_dict) #run this to see the dead pdf values
    alive_file_list=['AliveObjectXYs1a.txt','AliveObjectXYs2a.txt','AliveObjectXYs3a.txt']
    #alive_file_list = ['1-6-25_1_ObjectXYs.txt','1-6-25_2_ObjectXYs.txt','1-6-25_3_ObjectXYs.txt','1-6-25_4_ObjectXYs.txt','1-6-25_5_ObjectXYs.txt','1-6-25_6_ObjectXYs.txt','1-6-25_7_ObjectXYs.txt','1-6-25_8_ObjectXYs.txt','1-6-25_9_ObjectXYs.txt','1-6-25_10_ObjectXYs.txt','1-6-25_11_ObjectXYs.txt']
    #alive_file_list = ['1-3-25_1_ObjectXYs.txt','1-3-25_2_ObjectXYs.txt','1-3-25_3_ObjectXYs.txt','1-3-25_4_ObjectXYs.txt','1-3-25_5_ObjectXYs.txt','1-3-25_6_ObjectXYs.txt','1-3-25_7_ObjectXYs.txt','1-3-25_8_ObjectXYs.txt','1-3-25_9_ObjectXYs.txt','1-3-25_10_ObjectXYs.txt','1-3-25_11_ObjectXYs.txt','1-3-25_12_ObjectXYs.txt','1-3-25_13_ObjectXYs.txt','1-3-25_14_ObjectXYs.txt','1-3-25_15_ObjectXYs.txt']
    #alive_file_list = ['12-27-24_1_ObjectXYs.txt','12-27-24_2_ObjectXYs.txt','12-27-24_3_ObjectXYs.txt','12-27-24_4_ObjectXYs.txt','12-27-24_5_ObjectXYs.txt','12-27-24_6_ObjectXYs.txt','12-27-24_8_ObjectXYs.txt','12-27-24_9_ObjectXYs.txt','12-27-24_10_ObjectXYs.txt','12-27-24_11_ObjectXYs.txt','12-27-24_12_ObjectXYs.txt','12-27-24_13_ObjectXYs.txt','12-27-24_14_ObjectXYs.txt','12-27-24_15_ObjectXYs.txt','12-27-24_16_ObjectXYs.txt']


    alive_obs = load_observations(alive_file_list)
    #print(len(alive_obs))
    alive_grid_displacements=grid_by_grid_displacement_observation(alive_obs,5,4128,2196) 
    alive_grid_stats=grid_covariance_calculate(alive_grid_displacements,"alive")
    #print_grid_stats(alive_grid_stats)
    alive_pdf_all_dict=calculate_pdf_all_by_displacements(alive_obs,dead_grid_stats,4128,2196)
    dead_log_pdf_dict,alive_log_pdf_dict=get_log_dictionary(dead_pdf_all_dict,alive_pdf_all_dict)
    all_log_pdf_dict=combine_dictionaries(dead_log_pdf_dict,alive_log_pdf_dict)
    #mismatching_pdf_observations(alive_obs,alive_pdf_all_dict)
    #print(alive_pdf_all_dict) #run this to see the alive pdf values
    #data frame related stuffs
    '''
    #alive_df = convert_dict_to_dataframe(alive_pdf_all_dict,0)
    #print(alive_df)
    #dead_df = convert_dict_to_dataframe(dead_pdf_all_dict,1)
    #print(dead_df)
    #total_df=combine_df_write_to_csv(alive_df,dead_df)
    #print(total_df)
    #alive_dead_thresholding(total_df)
    #alive_dead_thresholding_sequential(total_df)
    
    #train_df,test_df=prepare_data(total_df)
    # Display the results
    #print(f"Train set distribution:\n{train_df['type'].value_counts()}")
    #print(f"Test set distribution:\n{test_df['type'].value_counts()}")

    #evaluate_model_performance(train_df,test_df)
    '''
    
    
    #get flattened list for dictionary having only pdfs it was to check only:
    dead_pdf_list=get_pdf_value_list(dead_pdf_all_dict)
    alive_pdf_list=get_pdf_value_list(alive_pdf_all_dict)
    #print(len(dead_pdf_list))
    '''
    get_unique_values_of_pdfs(dead_pdf_all_dict)
    #print(len(alive_pdf_list))
    get_unique_values_of_pdfs(alive_pdf_all_dict)
    '''
    
    '''
    dead_grid_obs=grid_by_grid_observation(dead_obs,5,4128,2196)
    print(dead_grid_obs)
    alive_grid_obs=grid_by_grid_observation(alive_obs,5,4128,2196)
    print(alive_grid_obs)
    '''
    
    '''
    alive_grid_displacements=grid_by_grid_displacement_observation(alive_obs,5,4128,2196) 
    alive_grid_stats=grid_covariance_calculate(alive_grid_displacements)
    print_grid_stats(alive_grid_stats)
    '''
    
    #result plot related codes start here:
    #mean_covariance_plot(alive_grid_stats)
    #plot_cdf_line_side_by_side(dead_pdf_list, alive_pdf_list)
    #plot_cdf_line_with_log_side_by_side(dead_pdf_list, alive_pdf_list)
    #plot_cdf_line_overlay(dead_pdf_list, alive_pdf_list)
    #plot_cdf_min_max_normalized_line_overlay(dead_pdf_list, alive_pdf_list)
    #plot_cdf_zscore_normalized_line_overlay(dead_pdf_list, alive_pdf_list)
    
    
    #plot_pdf_histogram_bins(alive_pdf_list,"Alive",0.5)   
    #plot_pdf_histogram_bins(dead_pdf_list,"Dead",0.5)
    #plot_pdf_overlay_histogram_bins(alive_pdf_list,dead_pdf_list)
    #filtered_overlay_histogram(dead_pdf_list,alive_pdf_list)
    #large_small_frequency_overlay_histogram(dead_pdf_list,alive_pdf_list)
    
    #make_collage()
    
    #dead_log_pdf_dict, alive_log_pdf_dict=get_log_dictionary(dead_pdf_all_dict,alive_pdf_all_dict)
    #get_thresholds_from_roc(all_log_pdf_dict)
   
    #best_threshold=thresholding_with_window_roc_curve(all_log_pdf_dict,2)
    #predicted_dict=predict_probabilities_dictionary_update(all_log_pdf_dict, best_threshold,2)
    
    #create_confusion_matrix(predicted_dict,best_threshold, "roc_curve","all ")
    prepare_data(dead_obs,alive_obs)
    
    
   
   
    
    
    