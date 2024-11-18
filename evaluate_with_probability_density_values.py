from scipy.stats import multivariate_normal
import numpy as np

from grid_by_grid_guassian_estimation import grid_by_grid_displacement_observation

def mismatching_pdf_observations(curr_obs,curr_pdf):
    i=0;
    for obj_id in curr_obs.keys() & curr_pdf.keys():
        obs_sizes=curr_obs[obj_id]
        pdf_sizes=curr_pdf[obj_id]
        
        #print(f"for {obj_id}: {obs_sizes,pdf_sizes}\n")
        #print(f"for {obj_id}: {len(obs_sizes),len(pdf_sizes)}\n")
        if(len(obs_sizes)-1!=len(pdf_sizes)):
            print(f"for {obj_id}: {len(obs_sizes),len(pdf_sizes)}")
            i+=1
    print(i)


def grid_by_grid_pdf(grid_stat, current_obj_dis):
    """
    this creates a list for the current object's pdf values with the help of grid statistics and object's displacements
    it traverses two lists one cell at a time extract's the mu,covariance and displacements in the particular cell
    Parameters:
    - grid_stat: 5*5 list of mu and covariance_matrix
    - curr_obj_dis: 5*5 list of displacement of one object
    Returns:
    - pdfs_for_cells: list of pdf values.
    """
    pdfs_for_cells=[]
    for i, (stats, obj_dis) in enumerate(zip(grid_stat,current_obj_dis)):
        for j, (stat_val,obj_pos) in enumerate(zip(stats, obj_dis)):
            mu=stat_val['mu']
            cov_matrix=stat_val['cov_matrix']
            curr_obj_dis_cord=obj_pos if obj_pos else []
            '''  
            print(f"at grid [{i}][{j}]")
            print(f"    mu: {mu}")              
            print(f"    cov_matrix:\n{cov_matrix}")
            print(f"    current displacement:\n{curr_obj_dis_cord}")
            '''
            
            if(len(curr_obj_dis_cord)>0):
                curr_dis_arr = np.array(curr_obj_dis_cord)
                mvn = multivariate_normal(mean=mu, cov=cov_matrix)
                curr_pdf_values=mvn.pdf(curr_dis_arr)
                curr_pdf_values = np.atleast_1d(curr_pdf_values)#converting to 1d array
                
                pdfs_for_cells.extend(curr_pdf_values)
                
                #print(f" calculated with numpy array:{curr_pdf_values.shape} {curr_pdf_values}")
    return pdfs_for_cells
    
def calculate_pdf_all_by_displacements(obs,grid_stats,max_x,max_y):
    """
    this creates a dictionary for all the pdf value for the displacements
    it takes one object observation at one time sends the dictionary (current_item_dict)to grid_by_grid_displacement_observation function to get the displacements of the objects
    take the returned 5*5 list (curr_obj_dis) calculates the the object's pdf in all the cells it has displacement
    Parameters:
    - obs: dictionary containing {object id: [(frame,x_coord,y_coord)]}
    - grid_stats: M*N list of mu and covariance matrices
    - max_x : x_coordinate maximum range
    - max_y : y_coordinate maximum range
    Returns:
    - pdf_all_dict: dictionary {object_id: [pdf values]}. Each pdf value
      corresponds to one displacement.
    """
    pdf_all_dict = {}
    i=0
    
    MAX_X=max_x
    MAX_Y=max_y
    grid_squares=len(grid_stats)
   
    for obj_id, obj_cord_list in obs.items():
        # Create a dictionary with the current object's ID and coordinates list
        current_item_dict = {obj_id: obj_cord_list}
        #print(f"from calculate all displacements functions: current object id: {obj_id}")
        
        # Compute displacement observations for the current object
        curr_obj_dis = grid_by_grid_displacement_observation(current_item_dict,grid_squares,MAX_X,MAX_Y)
        
        # Calculate Pdfs by grid for the current object's displacement observations
        pdfs_by_grid = grid_by_grid_pdf(grid_stats, curr_obj_dis)
        #print(pdfs_by_grid)
        # Store the result in dead_pdf_all_dict
        pdf_all_dict[obj_id] = pdfs_by_grid
        
    return pdf_all_dict

def get_pdf_value_list(curr_pdf):
    """
    this creates a list for all the pdf value for the displacements alive or dead
    it takes one object observation at one time sends the dictionary (current_item_dict)to grid_by_grid_displacement_observation function to get the displacements of the objects
    take the returned 5*5 list (curr_obj_dis) calculates the the object's pdf in all the cells it has displacement
    Parameters:
    - curr_pdf: dictionary containing {object id: (pdf_values_for displacements)}
    Returns:
    - pdf_values: list of all the pdf values alive or dead.
    """
    pdf_values=[]
    
    for val_list in curr_pdf.values():
        pdf_values.extend(val_list)
        
    return pdf_values


