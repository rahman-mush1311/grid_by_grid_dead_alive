import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from scipy.stats import multivariate_normal

def plot_pdf_histogram_bins(curr_pdf_list,object_type):
    
    log_values = np.log10(curr_pdf_list)
    bin_width = 0.5  # Each bin will cover a range of 
    min_value = min(log_values)
    max_value = max(log_values)
    bins = np.arange(min_value, max_value + bin_width, bin_width)  # Define bins with the specified width
    
    #print(min_value,max_value,len(bins))
    # Set up the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(log_values, bins=bins, color='skyblue', edgecolor='black')
    plt.xlabel("Log of Pdfs")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Log of {object_type} Pdfs Width =0.5")

    # Save the histogram as a PNG file with a higher resolution
    plt.savefig("log_value_occurrences_dead_histogram_bins.png", format='png', dpi=300)

    # Display the plot
    plt.show()

    
def plot_pdf_overlay_histogram_bins(alive_pdf_list,dead_pdf_list):
    
    alive_log_values = np.log10(alive_pdf_list)
    dead_log_values = np.log10(dead_pdf_list)
    
    bin_width = 0.5  # Each bin will cover a range of 
    min_value = min(min(alive_log_values),min(dead_log_values))
    max_value = max(max(alive_log_values),max(dead_log_values))
    bins = np.arange(min_value, max_value, bin_width)  # Define bins with the specified width
    
    #print(min_value,max_value,len(bins))
    # Set up the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(alive_log_values, bins=bins, alpha=0.5, label="Alive Pdfs", color="blue", edgecolor='black')
    plt.hist(dead_log_values, bins=bins, alpha=0.5, label="Dead Pdfs", color="red", edgecolor='black')
    
    plt.xlabel("Log of Pdfs")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Log of Alive & Dead Pdfs Width =0.5")

    # Save the histogram as a PNG file with a higher resolution
    plt.savefig("alive_dead_value_occurrences_histogram_bins.png", format='png', dpi=300)

    # Display the plot
    plt.show()

def object_plotting_with_pdf(pdf_with_cord):
    num_objects = len(pdf_with_cord)
    colors = plt.cm.viridis(np.linspace(0, 1, num_objects))  # Using viridis colormap for distinct colors

    # Plot each point, assigning a color from the color map
    plt.figure(figsize=(10, 6))

    for idx, (object_id, points) in enumerate(pdf_with_cord.items()):
        x_values = [p[0] for p in points]
        y_values = [p[1] for p in points]
        pdf_values = [p[2] for p in points]
        
        plt.plot(x_values, y_values, label=object_id, color=colors[idx], marker='o', linestyle='-', markersize=5)
    

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Dead Object's Scatter Plot With Pdf's")
    plt.legend(title="Object ID")
    plt.grid(True)
    #plt.savefig('dead object visuilizations')
    # Show the plot
    plt.show()
    
def contour_plot(grid_stat,grid_dis,grid_obs):

    pdfs_with_cords=[]
    for i, (stats, obj_dis,obj_cord) in enumerate(zip(grid_stat,grid_dis,grid_obs)):
        for j, (stat_val,obj_dis_pos,obj_pos) in enumerate(zip(stats, obj_dis,obj_cord)):
            mu=stat_val['mu']
            cov_matrix=stat_val['cov_matrix']
            curr_obj_dis_cord=obj_dis_pos if obj_dis_pos else []
            curr_obj_pos_cord=obj_pos if obj_pos else []
            
             
            #print(f"at grid [{i}][{j}]")
            #print(f"    mu: {mu}")              
            #print(f"    cov_matrix:\n{cov_matrix}")
            #print(f"displacements list lenght{len(curr_obj_dis_cord)} cordinates list {len(curr_obj_pos_cord),type(curr_obj_dis_cord)}, ")
            for coords in curr_obj_pos_cord:
                x,y =coords
                print(f"from visulize {x},{y}")
            if(len(curr_obj_dis_cord)>0 and len(curr_obj_pos_cord)):
                #print(f"    current displacement:\n{curr_obj_dis_cord}")
                #print(f"    current origina cordinate:\n{curr_obj_pos_cord}")
                curr_dis_arr = np.array(curr_obj_dis_cord)
                curr_obs_arr = np.array(curr_obj_pos_cord)
                #print(curr_dis_arr.shape,curr_obj_pos_cord)
            
            
            if(len(curr_obj_dis_cord)>0):
                mvn = multivariate_normal(mean=mu, cov=cov_matrix)
                curr_pdf_values=mvn.pdf(curr_dis_arr)
                
                    
    #print(len(pdfs_with_cords))          
    return pdfs_with_cords
