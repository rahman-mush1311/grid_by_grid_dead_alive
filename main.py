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
from grid_by_grid_guassian_estimation import load_observations,grid_by_grid_displacement_observation,grid_covariance_calculate,print_grid_stats
from evaluate_with_probability_density_values import grid_by_grid_pdf,calculate_pdf_obj,calculate_pdf_all_by_displacements,get_pdf_value_list
from result_visualization import plot_pdf_histogram_bins,plot_pdf_overlay_histogram_bins
# Main code execution
if __name__ == "__main__":
    
    #following 4 functions calculates grid by grid mu & covariance matrix
    dead_obs = load_observations('ObjectXYs.txt') 
    dead_grid_displacements=grid_by_grid_displacement_observation(dead_obs)
    dead_grid_stats=grid_covariance_calculate(dead_grid_displacements)
    print_grid_stats(dead_grid_stats) #run this to see the mu & sigma in formatted way

    #in below statement calculates the all dead object's probability density value of the displacements
    dead_pdf_all_dict=calculate_pdf_all_by_displacements(dead_obs,dead_grid_stats)
    #print(dead_pdf_all_dict) #run this to see the dead pdf values
    alive_obs = load_observations('AliveObjectXYs.txt')
    alive_pdf_all_dict=calculate_pdf_all_by_displacements(alive_obs,dead_grid_stats)
    #print(alive_pdf_all_dict) #run this to see the alive pdf values
    
    '''
    result plot related codes start here:
    '''
    alive_pdf_list=get_pdf_value_list(alive_pdf_all_dict)
    dead_pdf_list=get_pdf_value_list(dead_pdf_all_dict)
    #plot_pdf_histogram_bins(alive_pdf_list,"Alive")   
    #plot_pdf_histogram_bins(dead_pdf_list,"Dead")
    plot_pdf_overlay_histogram_bins(alive_pdf_list,dead_pdf_list)
