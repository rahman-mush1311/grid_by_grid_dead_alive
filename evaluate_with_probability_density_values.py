from scipy.stats import multivariate_normal

from grid_by_grid_guassian_estimation import grid_by_grid_displacement_observation

def calculate_pdf_obj(curr_obj_pos,mu,cov_matrix):
    """
    this creates a list for the a object's pdf values for one grid cell
    it traverses the list of all the displacements of one object in one grid cell calculates the multivariate_normal using numpy library functions
    Parameters:
    - curr_obj_pos: a list of displacement of one object on a particular grid
    - mu: list of means for dx,dy 
    - covariance: 2*2 list 
    Returns:
    - pdfs_for_obj: list of pdf values for one object's one grid cell.
    """
    
    pdfs_for_obj=[]    
    for point_to_evaluate in curr_obj_pos:
        #print(f"from calculate pdf of objects function: {point_to_evaluate}\n")
        pdf_value = multivariate_normal.pdf(point_to_evaluate, mean=mu, cov=cov_matrix)
        pdfs_for_obj.append(pdf_value)
        #print(f"pdf value is: {pdf_value}, list is: {pdfs_for_obj}\n")
    return pdfs_for_obj


def grid_by_grid_pdf(grid_stat, current_obj_dis):
    """
    this creates a list for the current object's pdf values with the help of grid statistics and object's displacements
    it traverses two lists one cell at a time extract's the mu,covariance and displacements in the particular cell
    Parameters:
    - grid_stat: 5*5 list of mu and covariance_matrix
    - curr_obj_dis: 5*5 list of displacement of one object
    Returns:
    - flatten_pdfs: list of pdf values.
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
                pdfs=calculate_pdf_obj(curr_obj_dis_cord,mu,cov_matrix)
                pdfs_for_cells.append(pdfs)
                #print(pdfs)
    flatten_pdfs=[pdfs for sublist in pdfs_for_cells for pdfs in sublist]
    return flatten_pdfs
    
def calculate_pdf_all_by_displacements(obs,grid_stats):
    """
    this creates a dictionary for all the pdf value for the displacements
    it takes one object observation at one time sends the dictionary (current_item_dict)to grid_by_grid_displacement_observation function to get the displacements of the objects
    take the returned 5*5 list (curr_obj_dis) calculates the the object's pdf in all the cells it has displacement
    Parameters:
    - obs: dictionary containing {object id: [(frame,x_coord,y_coord)]}
    - grid_stats: M*N list of mu and covariance matrices
    Returns:
    - pdf_all_dict: dictionary {object_id: [pdf values]}. Each pdf value
      corresponds to one displacement.
    """
    pdf_all_dict = {}
    i=0
    for obj_id, obj_cord_list in obs.items():
        # Create a dictionary with the current object's ID and coordinates list
        current_item_dict = {obj_id: obj_cord_list}
        #print(f"from calculate all displacements functions: current object id: {obj_id}")
        
        # Compute displacement observations for the current object
        curr_obj_dis = grid_by_grid_displacement_observation(current_item_dict)
        
        # Calculate Pdfs by grid for the current object's displacement observations
        pdfs_by_grid = grid_by_grid_pdf(grid_stats, curr_obj_dis)
        
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


