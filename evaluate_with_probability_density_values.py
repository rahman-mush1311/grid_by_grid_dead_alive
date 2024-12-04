from scipy.stats import multivariate_normal
import numpy as np
import pandas as pd
import os
import logging
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay,accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

from grid_by_grid_guassian_estimation import grid_by_grid_displacement_observation,grid_by_grid_observation

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
                pdfs=calculate_pdf_obj(curr_obj_dis_cord,mu,cov_matrix)
                '''
                curr_dis_arr = np.array(curr_obj_dis_cord)
                mvn = multivariate_normal(mean=mu, cov=cov_matrix)
                curr_pdf_values=mvn.pdf(curr_dis_arr)
                curr_pdf_values = np.atleast_1d(curr_pdf_values)#converting to 1d array               
                pdfs_for_cells.extend(curr_pdf_values)
                #contains_zero = np.any(pdfs == 0.00)
                #print("Contains zero:", contains_zero)
                '''
                pdfs_for_cells.append(pdfs)
                #print(pdfs)
    flatten_pdfs=[pdfs for sublist in pdfs_for_cells for pdfs in sublist]
    return flatten_pdfs
    #return pdfs_for_cells

def grid_by_grid_pdf_obs_dis(grid_stat, current_obj_dis,curr_obj_obs):
    """
    this creates a list for the current object's pdf values with the help of grid statistics and object's displacements
    it traverses three lists one cell at a time extract's the mu,covariance ,displacements and actual coordinates in the particular cell
    Parameters:
    - grid_stat: 5*5 list of mu and covariance_matrix
    - curr_obj_dis: 5*5 list of displacement of one object
    - curr_obj_obs: 5*5 list of the coordinate of one object
    Returns:
    - pdfs_obs_dis: list of (cooridantes, displacements, pdf corresponding values).
    """
    pdfs_obs_dis=[]
    
    for i, (stats, obj_dis,obj_cord) in enumerate(zip(grid_stat,current_obj_dis,curr_obj_obs)):
        for j, (stat_val,obj_dis_pos,obj_pos) in enumerate(zip(stats, obj_dis,obj_cord)):
            mu=stat_val['mu']
            cov_matrix=stat_val['cov_matrix']
            curr_obj_dis_cord=obj_dis_pos if obj_dis_pos else []
            curr_obj_pos_cord=obj_pos if obj_pos else []
            
            #print(f"at grid [{i}][{j}]")
            #print(f"    mu: {mu}")              
            #print(f"    cov_matrix:\n{cov_matrix}")
            #print(f"displacements list lenght{len(curr_obj_dis_cord)} cordinates list {len(curr_obj_pos_cord),type(curr_obj_dis_cord)}, ")
            if(len(curr_obj_dis_cord)>0 and len(curr_obj_pos_cord)):
                #print(f"    current displacement:\n{curr_obj_dis_cord}")
                #print(f"    current origina cordinate:\n{curr_obj_pos_cord}")
                curr_dis_arr = np.array(curr_obj_dis_cord)
                curr_obs_arr = np.array(curr_obj_pos_cord)
                #print(curr_dis_arr.shape,curr_obj_pos_cord)
            
            
            if(len(curr_obj_dis_cord)>0):
                mvn = multivariate_normal(mean=mu, cov=cov_matrix)
                curr_pdf_values=mvn.pdf(curr_dis_arr)
                #print(f"checking sizes: {type(curr_pdf_values)},{curr_obs_arr.shape}")
                curr_pdf_values = np.atleast_1d(curr_pdf_values)
                #pdfs_for_cells.extend(curr_pdf_values)
                #print(curr_dis_arr.shape,curr_obs_arr.shape,curr_pdf_values.shape)
                if (curr_dis_arr.shape[0]==curr_obs_arr.shape[0] and curr_obs_arr.shape[0]==curr_pdf_values.shape[0]):
                    for row1,row2,pdf_for_rows in zip(curr_dis_arr,curr_obs_arr,curr_pdf_values):
                        dx,dy=row1
                        x,y=row2
                        #print(x,y,dx,dy,pdf_for_rows)
                        pdfs_obs_dis.append((x,y,dx,dy,pdf_for_rows))
                    
    return pdfs_obs_dis   
        
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
    - pdf_all_dict: dictionary {object_id: [x,y,dx,dy,pdf value]}. Each pdf value
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
        
        # Compute displacements & observations for the current object
        curr_obj_dis = grid_by_grid_displacement_observation(current_item_dict,grid_squares,MAX_X,MAX_Y)
        curr_obj_obs = grid_by_grid_observation(current_item_dict,grid_squares,MAX_X,MAX_Y)
        
        # Calculate Pdfs by grid for the current object's displacement and their track of their observations and displacements   
        obs_dis_pdfs = grid_by_grid_pdf_obs_dis(grid_stats, curr_obj_dis,curr_obj_obs)
        #print(obs_dis_pdfs)
        pdf_all_dict[obj_id] = obs_dis_pdfs
        '''
        #pdfs_by_grid=grid_by_grid_pdf(grid_stats, curr_obj_dis)
        # Store the result in pdf_all_dict
        pdf_all_dict[obj_id] = pdfs_by_grid
        '''
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

def get_unique_values_of_pdfs(curr_pdfs):
   
    unique_pdfs = set()

    for values in curr_pdfs.values():
        unique_pdfs.update(values) 
        
    print(len(unique_pdfs))
    return
    
    
def alive_dead_thresholding(df):

    df = df[df["pdf"] > 0].copy()
    df["log_pdf"] = df["pdf"].apply(lambda x: np.log(x) if x > 0 else None)
    
   
    fpr, tpr, thresholds = roc_curve(df["type"], df["log_pdf"])  # True labels, and log probabilities
    roc_auc = auc(fpr, tpr)
    
    print(len(thresholds))
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)  # Index of the optimal threshold
    optimal_threshold = thresholds[optimal_idx]

    # Print the best threshold and corresponding metrics
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print(f"True Positive Rate (TPR): {tpr[optimal_idx]:.4f}")
    print(f"False Positive Rate (FPR): {fpr[optimal_idx]:.4f}")
    print(f"Youden's J Statistic: {youden_j[optimal_idx]:.4f}")
    
    threshold = optimal_threshold
    df["predicted_type"] = (df["log_pdf"] >= threshold).astype(int) #the boolean is true if log_pdf is greater than threshold else false
        
    # Evaluate Performance with Confusion Matrix
    cm = confusion_matrix(df["type"], df["predicted_type"])
    labels = ["Alive (0)", "Dead (1)"]

    # Convert confusion matrix to a DataFrame with row and column names
    cm_df = pd.DataFrame(cm, index=[f"True {label}" for label in labels],
                     columns=[f"Predicted {label}" for label in labels])

    # Print confusion matrix
    print("Confusion Matrix:")
    print(cm_df)
        
    # Calculate metrics
    accuracy = accuracy_score(df["type"], df["predicted_type"])
   
    precision = precision_score(df["type"], df["predicted_type"], zero_division=1)
    recall = recall_score(df["type"], df["predicted_type"], zero_division=1)
    f1 = f1_score(df["type"], df["predicted_type"], zero_division=1)
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}") 
    
    #show the ROC-AUC curve
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")  # Diagonal line
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', label=f"Optimal Threshold = {optimal_threshold:.2f}") #optimal threshold                    
    plt.xlabel("False Positive (Dead) Rate")
    plt.ylabel("True Positive (Dead) Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    #plt.savefig("ROC curve by using un")
    plt.show()
    
def alive_dead_thresholding_sequential(df):

    df = df[df["pdf"] > 0].copy()
    df["log_pdf"] = df["pdf"].apply(lambda x: np.log(x) if x > 0 else None)
    
    #print(len(df))
    
    df = df.reset_index(drop=True) #since dropping rows with 0.00 pdf values need to reset them so that it becomes sequential
    
    fpr, tpr, thresholds = roc_curve(df["type"], df["log_pdf"])  # True labels, and log probabilities
    roc_auc = auc(fpr, tpr)
    
    print(len(thresholds))
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)  # Index of the optimal threshold
    optimal_threshold = thresholds[optimal_idx]

    # Print the best threshold and corresponding metrics
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print(f"True Positive Rate (TPR): {tpr[optimal_idx]:.4f}")
    print(f"False Positive Rate (FPR): {fpr[optimal_idx]:.4f}")
    print(f"Youden's J Statistic: {youden_j[optimal_idx]:.4f}")
    
    threshold = optimal_threshold
    df["predicted_type"] = 0
    
    log_pdf_col_index = df.columns.get_loc("log_pdf")
    obj_id_col_index = df.columns.get_loc("obj_id")
    #print(log_pdf_col_index)
    # Iterate through rows to check conditions for sequential rows
    for idx in range(len(df) - 1):  # Exclude the last row to avoid index out of bounds
        # Check if current and next row satisfy the threshold and share the same object_id
        if idx+1 in df.index:
            curr_pdf,curr_obj_id=df.iloc[idx,log_pdf_col_index],df.iloc[idx,obj_id_col_index]
            next_pdf,next_obj_id=df.iloc[idx+1,log_pdf_col_index],df.iloc[idx+1,obj_id_col_index]
            if curr_pdf >=threshold and next_pdf>=threshold and curr_obj_id==next_obj_id:
                df.loc[idx, "predicted_type"] = 1
            else:
                df.loc[idx, "predicted_type"] = 0
        else:
            logging.error(f"the dataframe is missing row {idx+1}")
     
    # Evaluate Performance with Confusion Matrix
    cm = confusion_matrix(df["type"], df["predicted_type"])
    labels = ["Alive (0)", "Dead (1)"]
    
    # Convert confusion matrix to a DataFrame with row and column names
    cm_df = pd.DataFrame(cm, index=[f"True {label}" for label in labels],
                     columns=[f"Predicted {label}" for label in labels])
    # Print confusion matrix
    print("Confusion Matrix:")
    print(cm_df)
    
    # Calculate metrics
    accuracy = accuracy_score(df["type"], df["predicted_type"])
   
    precision = precision_score(df["type"], df["predicted_type"], zero_division=1)
    recall = recall_score(df["type"], df["predicted_type"], zero_division=1)
    f1 = f1_score(df["type"], df["predicted_type"], zero_division=1)
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}") 

   
    
    


