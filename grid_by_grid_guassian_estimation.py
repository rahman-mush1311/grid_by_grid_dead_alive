import re
import collections
import math
import numpy as np

def load_observations(filename):
    """
    this function processes the input file parses it to required data (object id, frame, x,y coordinates)
    
    Parameters:
    - filename: data file containing object and their coordinates and frame along with other description
    
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
    with open(filename) as object_xys:
        for line in object_xys:
            m = pattern.match(line)
            if m:
                obj_id = int(m.group('object_id'))
                frame = m.group('frame')
                cX=m.group('x')
                cY=m.group('y')
                
                observations[int(m.group('object_id'))].append((int(m.group('frame')), int(m.group('x')), int(m.group('y'))))
                
    return observations
 

def grid_by_grid_displacement_observation(curr_obs):
    """
    this function creates a 5*5 list (grid_obs) by processing a dictionary of observations where each grid cell contains the displacements
    the calculation of displacement across 2 axes is displacmenet along dx= x2-x1; dy=y2-y1
    and the cell where the displacement belongs is calculate by diving the x1 cord/maximum_of_x_cords * searching cell(grid_squares)
    
    
    Parameters:
    - curr_obs: dictionary containing object's observations(object id: (frame,x_cordinate,y_coordinate))
    
    Returns:
    - returns a 5*5 list of the displacements(dx,dy).
    """
    MAX_X=2000
    MAX_Y=4000
    
    GRID_SQUARES = 5

    grid_obs = [[[] for _ in range(GRID_SQUARES)] for _ in range(GRID_SQUARES)]
    #print(obs) 
    for obj_id, obs in curr_obs.items():
        #print(f"current obj is: {obj_id}:")
        #loop_count =0
        for j in range(len(obs)-1):
            #print(f"j is: {j}")
            #loop_count+=1
            x1=obs[j][1]
            y1=obs[j][2]
            x2=obs[j+1][1]
            y2=obs[j+1][2]

            x1_frame=obs[j][0]
            x2_frame=obs[j+1][0]
            
            x1_row=math.floor(x1/4000*GRID_SQUARES)
            y1_col=math.floor(y1/2000*GRID_SQUARES)

            x2_row=math.floor(x2/4000*GRID_SQUARES)
            y2_col=math.floor(y2/2000*GRID_SQUARES)
            #print(f"current cords are: {x1,y1}, {x2,y2} and row,col is: {x1_row,y1_col}")
            
            if(0<= x1_row <GRID_SQUARES and 0<= y1_col <GRID_SQUARES):
                grid_pos=grid_obs[x1_row][y1_col]
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
                        
        #assert loop_count == len(obs)-1, f"Loop count ({loop_count}) does not match list size ({len(obs)})"
        #print(f"The loop ran the same number of times as the list size, loop count is: {loop_count}")
        #print(missing_grid_obs)           
    return grid_obs

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
    covar=[]
    GRID_SQUARES = 5
    grid_stats = [[None for _ in range(GRID_SQUARES)] for _ in range(GRID_SQUARES)]
    
    for i in range(GRID_SQUARES):
        for j in range(GRID_SQUARES):
            if(len(grid_displacements[i][j])>1):
                dxdy_items = np.array(grid_displacements[i][j])
                #print(f" for {i,j} cell displacements are {dxdy_items}, {dxdy_items.shape}")
                #print(f" at {i,j} cell")
                mu = np.mean(dxdy_items, axis=0)
                #print(mu,mu.shape)
                
                cov_matrix = np.cov(dxdy_items.T)
                #print(cov_matrix,cov_matrix.shape)
                
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
            mu=col_item['mu']
            cov_matrix=col_item['cov_matrix']
            print(f"grid[{i}][{j}]: ")               
            print(f"    mu: {mu}")              
            print(f"    cov_matrix:\n{cov_matrix}")
 