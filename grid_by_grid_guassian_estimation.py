import re
import collections
import math
import numpy as np
import logging
import os
import pandas as pd

LOG_DIRECTORY = r"D:\RA work Fall2024\grid_by_grid_dead_alive\log_info"
LOG_FILE = "gaussian_estimation.log"

# Ensure the directory exists
os.makedirs(LOG_DIRECTORY, exist_ok=True)

# Full path to the log file
LOG_PATH = os.path.join(LOG_DIRECTORY, LOG_FILE)

# Configure logging globally
logging.basicConfig(
    level=logging.WARNING,  # Set the minimum level globally
    format="%(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Log to the console
        logging.FileHandler(LOG_PATH)  # Log to the specified file
    ]
)

def load_observations(filenames):
    """
    this function processes the input file parses it to required data (object id, frame, x,y coordinates)
    
    Parameters:
    - filenames: list of data files which contains object and their coordinates and frame along with other description
    
    Returns:
    - returns a dictionary of the observation (alive/dead).
    """
    
    pattern = re.compile(r'''
    \s*(?P<object_id>\d+),            # object_id (first number)
    \s*(?P<within_frame_id>\d+),       # within_frame_id (second number)
    \s*'(?P<file_path>[^']+)',              # file path (file location)
    \s*cX\s*=\s*(?P<x>\d+),                 # cX coordinate
    \s*cY\s*=\s*(?P<y>\d+),                 # cY coordinate
    \s*Frame\s*=\s*(?P<frame>\d+)           # Frame number
''', re.VERBOSE)
    
    observations = collections.defaultdict(list)
    for filename in filenames:
        with open(filename) as object_xys:
            for line in object_xys:
                m = pattern.match(line)
                if m:
                    obj_id = int(m.group('object_id'))
                    frame = m.group('frame')
                    cX=m.group('x')
                    cY=m.group('y')
                
                    observations[int(m.group('object_id'))].append((int(m.group('frame')), int(m.group('x')), int(m.group('y'))))

    # make sure the observations are in frame order
    
    for object_id in observations:
        observations[object_id].sort()
                
    return observations
 

def grid_by_grid_displacement_observation(curr_obs,grid_squares,max_x,max_y):
    """
    this function creates a 5*5 list (grid_dis) by processing a dictionary of observations where each grid cell contains the displacements
    the calculation of displacement across 2 axes is displacmenet along dx= x2-x1; dy=y2-y1
    and the cell where the displacement belongs is calculate by diving the x1 cord/maximum_of_x_cords * searching cell(grid_squares)
    
    
    Parameters:
    - curr_obs: dictionary containing object's observations(object id: (frame,x_cordinate,y_coordinate))
    - grid_square: range for grid search
    - max_x : x_coordinate maximum range
    - max_y : y_coordinate maximum range
    Returns:
    - returns a 5*5 list of the displacements(dx,dy).
    """
    
    # FIXME - make these parameters(w)
    MAX_X=max_x
    MAX_Y=max_y
    
    # FIXME - make this a parameter(w)
    GRID_SQUARES = grid_squares

    grid_dis = [[[] for _ in range(GRID_SQUARES)] for _ in range(GRID_SQUARES)]
    #print(obs) 
    for obj_id, obs in curr_obs.items():
        #print(f"current obj is: {obj_id}:")
        #loop_count =0
        for j in range(len(obs)-1):
            #print(f"j is: {j}")
            #loop_count+=1
            x1_frame, x1, y1 = obs[j]
            x2_frame, x2, y2 = obs[j+1]
            
            # FIXME - don't use 4000/2000 constants; use MAX_X and MAX_Y(w)
            x1_row=math.floor(x1/MAX_X*GRID_SQUARES)
            y1_col=math.floor(y1/MAX_Y*GRID_SQUARES)

            x2_row=math.floor(x2/MAX_X*GRID_SQUARES)
            y2_col=math.floor(y2/MAX_Y*GRID_SQUARES)
            #print(f"current cords are: {x1,y1}, {x2,y2} and row,col is: {x1_row,y1_col}")
            
            # FIXME - if the below "if" is not true, then we are losing data;(logging to a log file)
            # this should be an error (rather than silently ignored)
            if(0<= x1_row <GRID_SQUARES and 0<= y1_col <GRID_SQUARES):
                grid_pos=grid_dis[x1_row][y1_col]
                if(x1_frame==(x2_frame+1)):
                    dx=x2-x1
                    dy=y2-y1
                    grid_pos.append((dx,dy))
                    #print(f"current frame is: {x1_frame} , {dx,dy}")
                
                else:
                    frame_distance=x2_frame-x1_frame 
                    if (frame_distance!=0):
                        dx=(x2-x1)/frame_distance
                        dy=(y2-y1)/frame_distance
                        grid_pos.append((dx,dy))
                        #print(f"skipping frame is: {x1_frame} , {dx,dy}")
            else:
                #print(f"missing data for the grid, for grid[{x1_row}][{y1_col}] for {x1,y1}.")
                logging.error(f"missing data for the grid, for grid[{x1_row}][{y1_col}] for {x1,y1}.")
                
        #assert loop_count == len(obs)-1, f"Loop count ({loop_count}) does not match list size ({len(obs)})"
        #print(f"The loop ran the same number of times as the list size, loop count is: {loop_count}")          
    return grid_dis

def grid_covariance_calculate(grid_displacements):
    """
    this function creates a 5*5 list (grid_stats) from grid_displacements (5*5 list (dx,dy)) ,in the grid_stats each grid cell has their mu and covariance_matrix of size(2*2)
    it calculates by 1 grid cell at one time using numpy mean function and numpy cov function
    before the calculation it converts the list of displacement (in a particular cell) to a numpy array
    Parameters:
    - grid_displacements: grid_displacements (5*5 list where all the displacement's lie)
    
    Returns:
    - returns a 5*5 list of mu and covariance matrix.
    """

    # FIXME - make GRID_SQUARES into rows and columns, based on the shape of(w)
    # grid_displacements
    grid_stats = [[None for _ in range(len(row))] for row in grid_displacements]
    flag=False
    
    for i in range(len(grid_displacements)):
        for j in range(len(grid_displacements[i])):
            # FIXME - this if statement should always be true; if it's not, we don't have enough data to estimate mu/sigma for this cell. (w)
            # In fact, we should have at least 10, preferably 30 examples per cell.
            # If we don't, we don't have enough data - and that should at least
            # be a warning, if not an error.(w)
            if(len(grid_displacements[i][j])>1):
                if(len(grid_displacements[i][j])<30):
                    logging.warning(f"grid[{i}][{j}] has less than 30 observations.")
                dxdy_items = np.array(grid_displacements[i][j])
                #print(f" for {i,j} cell displacements are {dxdy_items}, {dxdy_items.shape}")
                #print(f" at {i,j} cell")
                mu = np.mean(dxdy_items, axis=0)
                #print(mu,mu.shape)
                
                cov_matrix = np.cov(dxdy_items.T)
                #print(cov_matrix,cov_matrix.shape)
                #logging.warning(f"not enough data to calculate mu & sigma.")
            else:
                flag=True
                logging.error(f"grid[{i}][{j}] doesnot enough data to calculate mu & sigma for grid[{i}][{j}].")
            # FIXME - this can set mu/cov_matrix based on a previous iteration
            # if the above if statement is false
            if flag==False:
                grid_stats[i][j] = {
                    'mu': mu,
                    'cov_matrix': cov_matrix
                }
                      
    #print(grid_stats)
    return grid_stats 
 
def print_grid_stats(grid_stat):
    """
    this function prints the mu and covariance by grid cells
    Parameters:
    - grid_stat: 5*5 list of mu and covariance_matrix
    
    Returns:
    - N/A.
    """
    for i, row_item in enumerate(grid_stat):
        for j, col_item in enumerate(row_item):
            if col_item is not None:
                mu=col_item['mu']
                cov_matrix=col_item['cov_matrix']
                print(f"grid[{i}][{j}]: ")               
                print(f"    mu: {mu}")              
                print(f"    cov_matrix:\n{cov_matrix}")
            else:
                print(f"grid[{i}][{j}]: None")

def get_unique_values_of_pdfs(curr_pdfs):
   
    unique_pdfs = set()

    for values in curr_pdfs.values():
        unique_pdfs.update(values) 
        
    #print(unique_pdfs)
    return unique_pdfs

def grid_by_grid_observation(curr_obs,grid_squares,max_x,max_y):
    """
    this function creates a 5*5 list (grid_obs) by processing a dictionary of observations where each grid cell contains the coordinates it is located
    and the cell where the displacement belongs is calculate by diving the x1 cord/maximum_of_x_cords * searching cell(grid_squares)
    
    
    Parameters:
    - curr_obs: dictionary containing object's observations(object id: (frame,x_cordinate,y_coordinate))
    - grid_square: range for grid search
    - max_x : x_coordinate maximum range
    - max_y : y_coordinate maximum range
    Returns:
    - returns a 5*5 list of the observations(x,y).
    """
    MAX_X=max_x
    MAX_Y=max_y
    
    GRID_SQUARES = grid_squares
    
    grid_obs = [[[] for _ in range(GRID_SQUARES)] for _ in range(GRID_SQUARES)]#grid by grid actual observations not displacements
    missing_grid_obs={}
    for obj_id, obs in curr_obs.items():
        for j in range(len(obs)-1):
            x1_frame,x1,y1=obs[j]
            x2_frame,x2,y2=obs[j+1]
            
            x1_row=math.floor(x1/MAX_X*GRID_SQUARES)
            y1_col=math.floor(y1/MAX_Y*GRID_SQUARES)

            x2_row=math.floor(x2/MAX_X*GRID_SQUARES)
            y2_col=math.floor(y2/MAX_Y*GRID_SQUARES)
            
            if(0<= x1_row <GRID_SQUARES and 0<= y1_col <GRID_SQUARES):
                grid_pos=grid_obs[x1_row][y1_col]
                if(x1_frame==(x2_frame+1)):
                    grid_pos.append((x1,y1))
                else:
                    frame_distance=x2_frame-x1_frame 
                    if (frame_distance!=0):
                        grid_pos.append((x1,y1))
            else:
                logging.error(f"missing data for the grid, for grid[{x1_row}][{y1_col}] for {x1,y1}.")
                
    #print(grid_obs)             
    return grid_obs
 

'''   
def convert_dict_to_dataframe(data_dict, data_type):
    # Create an empty list to hold the rows
    rows = []
    
    # Iterate through the dictionary
    for object_id, values in data_dict.items():
        for value in values:
            # Append each row, adding the object_id and data_type
            rows.append([object_id, *value, data_type])
    
    # Create a DataFrame
    columns = ["obj_id", "x", "y", "dx", "dy", "pdf", "type"]
    return pd.DataFrame(rows, columns=columns)
'''
 
