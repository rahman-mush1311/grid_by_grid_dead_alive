import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
import matplotlib as mpl
from scipy.stats import skew
mpl.rcParams['figure.max_open_warning'] = 100 
from PIL import Image
import os
import math

def plot_pdf_histogram_bins(curr_pdf_list,object_type,width_of_bin):
    """
    this function creates histogram of the object's log probability density values
    
    Parameters:
    - curr_pdf_list: list containing pdf values
    - object_type: string indicating alive/dead
    - width_of_bin: in which width bins will be created
    
    Returns:
    - N/A
    """
    curr_pdf_array = np.array(curr_pdf_list)
    '''
    # Check if any value is 0
    contains_invalid_values = np.any(curr_pdf_array <= 0)
    print(contains_invalid_values)
    '''
    log_values = np.log10(curr_pdf_array[curr_pdf_array > 0])
    bin_width = width_of_bin  # Each bin will cover a range of 
    min_value = min(log_values)
    max_value = max(log_values)
    bins = np.arange(min_value, max_value + bin_width, bin_width)  # Define bins with the specified width
    
    print(min_value,max_value,len(bins))
    hist_values, bin_edges = np.histogram(log_values, bins=bins)
    
    filtered_bins = [(bin_edges[i], bin_edges[i + 1]) for i in range(len(hist_values)) if hist_values[i] > 0]
    filtered_freqs = [freq for freq in hist_values if freq > 0]
    #print(filtered_bins)
    print(len(filtered_freqs))
    
    # Set up the histogram
    plt.figure(figsize=(10, 6))
    for i, (start, end) in enumerate(filtered_bins):
        plt.bar(
            x=start,
            height=filtered_freqs[i],
            width=bin_width,
            color='skyblue',
            edgecolor='black',
            align='center',
            label=f"Bin ({start:.2f}, {end:.2f})" if i == 0 else ""  # Add label for legend only once
        )

    plt.xlabel("Log of Pdfs")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Log of {object_type} Pdfs Width ={bin_width}")

    # Save the histogram as a PNG file with a higher resolution
    #plt.savefig(f"value_occurrences_histogram_bins_{object_type}_{bin_width}.png", format='png', dpi=300)

    # Display the plot
    plt.show()

    
def plot_pdf_overlay_histogram_bins(alive_pdf_list,dead_pdf_list):
    
    alive_pdf_arr=np.array(alive_pdf_list)
    alive_log_values = np.log10(alive_pdf_arr[alive_pdf_arr >0])
    dead_pdf_arr=np.array(dead_pdf_list)
    dead_log_values = np.log10(dead_pdf_arr[dead_pdf_arr >0])
    
    
    bin_width = 0.5  # Each bin will cover a range of 
    min_value = min(min(alive_log_values),min(dead_log_values))
    max_value = max(max(alive_log_values),max(dead_log_values))
    bins = np.arange(min_value, max_value, bin_width)  # Define bins with the specified width
    
    print(min_value,max_value,len(bins))
    # Set up the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(alive_log_values, bins=bins, alpha=0.5, label="Alive Pdfs", color="blue", edgecolor='black')
    plt.hist(dead_log_values, bins=bins, alpha=0.5, label="Dead Pdfs", color="red", edgecolor='black')
    
    plt.xlabel("Log of Pdfs")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Log of Alive & Dead Pdfs Width =0.5")

    # Save the histogram as a PNG file with a higher resolution
    #plt.savefig("overlay_value_occurrences_histogram_bins_0.5.png", format='png', dpi=300)

    # Display the plot
    plt.show()
    
def large_small_frequency_overlay_histogram(dead_pdf_list,alive_pdf_list):

    alive_pdf_arr=np.array(alive_pdf_list)
    alive_log_values = np.log10(alive_pdf_arr[alive_pdf_arr >0])
    dead_pdf_arr=np.array(dead_pdf_list)
    dead_log_values = np.log10(dead_pdf_arr[dead_pdf_arr >0])
    
    
    bin_width = 0.5  # Each bin will cover a range of 
    min_value = min(min(alive_log_values),min(dead_log_values))
    max_value = max(max(alive_log_values),max(dead_log_values))
    bins = np.arange(min_value, max_value, bin_width)  # Define bins with the specified width
    
    alive_hist_values, alive_bin_edges = np.histogram(alive_log_values, bins=bins)
    
    alive_filtered_bins = [(alive_bin_edges[i], alive_bin_edges[i + 1]) for i in range(len(alive_hist_values)) if alive_hist_values[i] > 0]
    alive_filtered_freqs = [freq for freq in alive_hist_values if freq > 0]
    #print(alive_filtered_bins)
    #print(alive_filtered_freqs)
    
    dead_hist_values, dead_bin_edges = np.histogram(dead_log_values, bins=bins)
    
    dead_filtered_bins = [(dead_bin_edges[i], dead_bin_edges[i + 1]) for i in range(len(dead_hist_values)) if dead_hist_values[i] > 0]
    dead_filtered_freqs = [freq for freq in dead_hist_values if freq > 0]
    
    threshold=min(np.mean(alive_filtered_freqs),np.mean(dead_filtered_freqs))
    print(threshold)
    
    bin_centers_1 = [b[0] for b in alive_filtered_bins]
    bin_centers_2 = [b[0]  for b in dead_filtered_bins]
    
    small_frequencies_alive = [freq for freq in alive_filtered_freqs if freq <= threshold]
    small_bins_alive = [bin_centers_1[i] for i, freq in enumerate(alive_filtered_freqs) if freq <= threshold]

    large_frequencies_alive = [freq for freq in alive_filtered_freqs if freq > threshold]
    large_bins_alive = [bin_centers_1[i] for i, freq in enumerate(alive_filtered_freqs) if freq > threshold]

    # Separate small and large frequencies for Dataset 2
    small_frequencies_dead = [freq for freq in dead_filtered_freqs if freq <= threshold]
    small_bins_dead = [bin_centers_2[i] for i, freq in enumerate(dead_filtered_freqs) if freq <= threshold]

    large_frequencies_dead = [freq for freq in dead_filtered_freqs if freq > threshold]
    large_bins_dead = [bin_centers_2[i] for i, freq in enumerate(dead_filtered_freqs) if freq > threshold]
    
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.bar(small_bins_alive, small_frequencies_alive, width=0.5, color='red', alpha=0.5, label='alive')
    plt.bar(small_bins_dead, small_frequencies_dead, width=0.5, color='green', alpha=0.5, label='dead', edgecolor='black')
    plt.title("Small Frequency Bins For Alive(R) & Dead(G)", fontsize=14)
    #plt.title("Small Frequency Bins For Dead", fontsize=14)
    plt.xlabel("Logarithmic Pdf Values", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    #plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Plot large frequencies
    plt.subplot(1, 2, 2)
    plt.bar(large_bins_alive, large_frequencies_alive, width=0.5, color='blue',alpha=0.5, label='alive', edgecolor='black')
    plt.bar(large_bins_dead, large_frequencies_dead, width=0.5, color='orange',alpha=0.5, label='dead', edgecolor='black')
    plt.title("Large Frequency Bins For Dead(O) & Alive (B)", fontsize=14)
    #plt.title("Large Frequency Bins For Dead", fontsize=14)
    plt.xlabel("Logarithmic Pdf Values", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    #plt.grid(axis='y', linestyle='--', alpha=0.7)
    #plt.savefig("dead & alive overlayed largest & smallest frequency")
    plt.tight_layout()
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
    
def mean_covariance_plot(grid_stat):
    # Step 1: Compute global range for all plots
    global_min_x, global_max_x = float('inf'), float('-inf')
    global_min_y, global_max_y = float('inf'), float('-inf')

    # Iterate through the grid to find the global range
    for row_item in grid_stat:
        for col_item in row_item:
            mu = col_item['mu']
            cov_matrix = col_item['cov_matrix']
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            width, height = 2 * np.sqrt(eigenvalues)
            max_range = max(width, height) * 1.5
            
            # Update global min and max values
            global_min_x = min(global_min_x, mu[0] - max_range)
            global_max_x = max(global_max_x, mu[0] + max_range)
            global_min_y = min(global_min_y, mu[1] - max_range)
            global_max_y = max(global_max_y, mu[1] + max_range)
            
    print(global_max_y,global_max_x)
    print(global_min_y,global_max_y)
    
    for i, row_item in enumerate(grid_stat):
        for j, col_item in enumerate(row_item):
            mu=col_item['mu']
            cov_matrix=col_item['cov_matrix']
            
            fig, ax = plt.subplots(figsize=(8, 8))

            # Plot the mean as a point
            ax.plot(mu[0], mu[1], 'ro', label="Mean")
            
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            # Sort eigenvalues and eigenvectors
            order = eigenvalues.argsort()[::-1]
            eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    
            # Calculate the angle of the ellipse
            angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

            # Width and height of the ellipse based on n_std standard deviations
            width, height = 2 * 1.0 * np.sqrt(eigenvalues)

            # Draw the ellipse
            ellipse_1std = Ellipse(xy=mu, width=width, height=height, angle=angle,edgecolor='blue', linestyle='--', linewidth=2, facecolor='none', label="1 Std Dev")
            ax.add_patch(ellipse_1std)
            
            ax.set_xlim(global_min_x, global_max_x)
            ax.set_ylim(global_min_y, global_max_y)

            # Set equal aspect ratio for both axes
            ax.set_aspect('equal', adjustable='datalim')
            
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.legend()
            ax.grid(True)
            ax.set_title(f"Visualization of Mean and Covariance Matrix as Ellipses For Alive Grid {[i]}{[j]}")
            #plt.axis('equal')
            #plt.savefig(f"dead_grid_stats_in_same_range[{i}][{j}]")
            plt.show()

def plot_cdf_line_side_by_side(dead_pdf_list, alive_pdf_list):
    # Calculate log values of pdfs
    #dead_log_values = np.log10(dead_pdf_list)
    #alive_log_values= np.log10(alive_pdf_list)
    
    # Step 1: Sort the probability density values
    dead_sorted_values = np.sort(dead_pdf_list)
    alive_sorted_values = np.sort(alive_pdf_list)
    # Step 2: Calculate the cumulative probabilities
    dead_cdf = np.arange(1, len(dead_sorted_values) + 1) / len(dead_sorted_values)
    alive_cdf = np.arange(1, len(alive_sorted_values) + 1) / len(alive_sorted_values)
    
    print("Range of PDF 1:", dead_sorted_values.min(), "to", dead_sorted_values.max())
    print("Range of PDF 2:", alive_sorted_values.min(), "to", alive_sorted_values.max())
    

    # Step 3: Plot the CDFs side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # CDF for the first set of probability density values
    axes[0].plot(dead_sorted_values, dead_cdf,color='blue')
    axes[0].set_title('CDF of Dead PDF', fontsize=14)
    axes[0].set_xlabel('Probability Density Values Dead', fontsize=12)
    axes[0].set_ylabel('Cumulative Probability', fontsize=12)
    axes[0].grid()

    # CDF for the second set of probability density values
    axes[1].plot(alive_sorted_values, alive_cdf,color='green')
    axes[1].set_title('CDF of Alive PDF ', fontsize=14)
    axes[1].set_xlabel('Probability Density Values Dead', fontsize=12)
    axes[1].set_ylabel('Cumulative Probability', fontsize=12)
    axes[1].grid()
    
    plt.savefig(f"dead_alive_cdf_line(n).png", format='png', dpi=300)
    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

def plot_cdf_line_with_log_side_by_side(dead_pdf_list, alive_pdf_list):
    # Calculate log values of pdfs
    dead_log_values = np.log10([x for x in dead_pdf_list if x > 0])
    alive_log_values= np.log10([x for x in alive_pdf_list if x > 0])
    
    # Step 1: Sort the probability density values
    dead_sorted_values = np.sort(dead_log_values)
    alive_sorted_values = np.sort(alive_log_values)
    # Step 2: Calculate the cumulative probabilities
    dead_cdf = np.arange(1, len(dead_sorted_values) + 1) / len(dead_sorted_values)
    alive_cdf = np.arange(1, len(alive_sorted_values) + 1) / len(alive_sorted_values)
    
    print("Range of PDF 1:", dead_sorted_values.min(), "to", dead_sorted_values.max())
    print("Range of PDF 2:", alive_sorted_values.min(), "to", alive_sorted_values.max())
    

    # Step 3: Plot the CDFs side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # CDF for the first set of probability density values
    axes[0].plot(dead_sorted_values, dead_cdf,color='blue')
    axes[0].set_title('CDF of Dead PDF', fontsize=14)
    axes[0].set_xlabel('Probability Density Values Dead', fontsize=12)
    axes[0].set_ylabel('Cumulative Probability', fontsize=12)
    axes[0].grid()

    # CDF for the second set of probability density values
    axes[1].plot(alive_sorted_values, alive_cdf,color='green')
    axes[1].set_title('CDF of Alive PDF ', fontsize=14)
    axes[1].set_xlabel('Probability Density Values Dead', fontsize=12)
    axes[1].set_ylabel('Cumulative Probability', fontsize=12)
    axes[1].grid()
    
    plt.savefig(f"dead_alive_cdf_line_log.png", format='png', dpi=300)
    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()
    
def plot_cdf_line_overlay(dead_pdf_list, alive_pdf_list):
    # Calculate log values of pdfs
    dead_log_values = np.log10([x for x in dead_pdf_list if x > 0])
    alive_log_values= np.log10([x for x in alive_pdf_list if x > 0])
    
    # Step 1: Sort the probability density values
    dead_sorted_values = np.sort(dead_log_values)
    alive_sorted_values = np.sort(alive_log_values)
    # Step 2: Calculate the cumulative probabilities
    dead_cdf = np.arange(1, len(dead_sorted_values) + 1) / len(dead_sorted_values)
    alive_cdf = np.arange(1, len(alive_sorted_values) + 1) / len(alive_sorted_values)
    
    plt.figure(figsize=(10, 6))
    plt.plot(dead_sorted_values, dead_cdf,label='dead_pdfs', marker='.', linestyle='none', color='blue')
    plt.plot(alive_sorted_values, alive_cdf,label='dead_pdfs', marker='.', linestyle='none', color='red')
    #plt.plot(sorted_values, cdf, color='blue') #for line
    plt.xlabel('Log Probability Density Values')
    plt.ylabel('Cumulative Probability')
    plt.title(f"Cumulative Distribution Function (CDF) For Dead(B) & Alive(R)")
    #plt.savefig(f"overlay_dead_alive_cdf_scatter_(n).png", format='png', dpi=300)
    plt.grid()
    plt.show()
    
def plot_cdf_min_max_normalized_line_overlay(dead_pdf_list, alive_pdf_list):
    # Calculate log values of pdfs
    dead_log_values = np.log10([x for x in dead_pdf_list if x > 0])
    alive_log_values= np.log10([x for x in alive_pdf_list if x > 0])
    '''
    # Step 1: Normalize and Offset the Probability Density Values
    norm_pdf_dead = (dead_pdf_list - min(dead_pdf_list)) / (max(dead_pdf_list) - min(dead_pdf_list))
    norm_pdf_alive = (alive_pdf_list - min(alive_pdf_list)) / (max(alive_pdf_list) - min(alive_pdf_list)) + 0.5  # Adding offset to avoid overlap
    '''
    # Step 1: Normalize and Offset the Log Probability Density Values
    norm_pdf_dead = (dead_log_values - min(dead_log_values)) / (max(dead_log_values) - min(dead_log_values))
    norm_pdf_alive = (alive_log_values - min(alive_log_values)) / (max(alive_log_values) - min(alive_log_values))   # Adding offset to avoid overlap
    # Step 2: Add an offset to the second dataset's sorted values
    dead_sorted_values = np.sort(norm_pdf_dead)
    alive_sorted_values = np.sort(norm_pdf_alive)
    
    # Step 3: Calculate the cumulative probabilities
    dead_cdf = np.arange(1, len(dead_sorted_values) + 1) / len(dead_sorted_values)
    alive_cdf = np.arange(1, len(alive_sorted_values) + 1) / len(alive_sorted_values)
    
    plt.figure(figsize=(10, 6))
    plt.plot(dead_sorted_values, dead_cdf,label='dead_pdfs', marker='.', linestyle='none', color='blue')
    plt.plot(alive_sorted_values, alive_cdf,label='dead_pdfs', marker='.', linestyle='none', color='red')
    #plt.plot(sorted_values, cdf, color='blue') #for line
    plt.xlabel('Log Probability Density Values(Min-Max Normalized)')
    plt.ylabel('Cumulative Probability')
    plt.title(f"Cumulative Distribution Function (CDF) For Dead(B) & Alive(R) Without Offset")
    #plt.savefig(f"overlay_log_dead_alive_cdf_scatter_min_max_noramlization_without_offset(n).png", format='png', dpi=300)
    plt.grid()
    plt.show()
   

def plot_cdf_zscore_normalized_line_overlay(dead_pdf_list, alive_pdf_list):
    # Calculate log values of pdfs
    dead_log_values = np.log10([x for x in dead_pdf_list if x > 0])
    alive_log_values= np.log10([x for x in alive_pdf_list if x > 0])
    
    # Step 1: Normalize and Offset the Probability Density Values
    zscore_pdf_dead = (dead_pdf_list - np.mean(dead_pdf_list)) / np.std(dead_pdf_list)
    zscore_pdf_alive = (alive_pdf_list - np.mean(alive_pdf_list)) / np.std(alive_pdf_list)  # Adding offset to avoid overlap
    
    # Step 2: Add an offset to the second dataset's sorted values
    dead_sorted_values = np.sort(zscore_pdf_dead)
    alive_sorted_values = np.sort(zscore_pdf_alive+0.5)
    
    # Step 3: Calculate the cumulative probabilities
    dead_cdf = np.arange(1, len(dead_sorted_values) + 1) / len(dead_sorted_values)
    alive_cdf = np.arange(1, len(alive_sorted_values) + 1) / len(alive_sorted_values)
    
    #print("Range of PDF 1:", dead_sorted_values.min(), "to", dead_sorted_values.max())
    #print("Range of PDF 2:", alive_sorted_values.min(), "to", alive_sorted_values.max())
    
    plt.figure(figsize=(10, 6))
    plt.plot(dead_sorted_values, dead_cdf,label='dead_pdfs', marker='.', linestyle='none', color='blue')
    plt.plot(alive_sorted_values, alive_cdf,label='dead_pdfs', marker='.', linestyle='none', color='red')
    #plt.plot(sorted_values, cdf, color='blue') #for line
    plt.xlabel('Probability Density Values(Normalized With Zscore)')
    plt.ylabel('Cumulative Probability')
    plt.title(f"Cumulative Distribution Function (CDF) For Dead(B) & Alive(R) Offset 0.5")
    #plt.savefig(f"overlay_dead_alive_cdf_scatter_zscore_noramlization with offset.png", format='png', dpi=300)
    plt.grid()
    plt.show()

def filtered_overlay_histogram(dead_pdf_list,alive_pdf_list):

    alive_pdf_arr=np.array(alive_pdf_list)
    alive_log_values = np.log10(alive_pdf_arr[alive_pdf_arr >0])
    dead_pdf_arr=np.array(dead_pdf_list)
    dead_log_values = np.log10(dead_pdf_arr[dead_pdf_arr >0])
    
    
    bin_width = 1  # Each bin will cover a range of 
    min_value = min(min(alive_log_values),min(dead_log_values))
    max_value = max(max(alive_log_values),max(dead_log_values))
    bins = np.arange(min_value, max_value, bin_width)  # Define bins with the specified width
    
    alive_hist_values, alive_bin_edges = np.histogram(alive_log_values, bins=bins)
    
    alive_filtered_bins = [(alive_bin_edges[i], alive_bin_edges[i + 1]) for i in range(len(alive_hist_values)) if alive_hist_values[i] > 0]
    alive_filtered_freqs = [freq for freq in alive_hist_values if freq > 0]
    #print(alive_filtered_bins)
    #print(alive_filtered_freqs)
    
    dead_hist_values, dead_bin_edges = np.histogram(dead_log_values, bins=bins)
    
    dead_filtered_bins = [(dead_bin_edges[i], dead_bin_edges[i + 1]) for i in range(len(dead_hist_values)) if dead_hist_values[i] > 0]
    dead_filtered_freqs = [freq for freq in dead_hist_values if freq > 0]
    #print(dead_filtered_bins)
    #print(dead_filtered_freqs)
    bin_centers_1 = [b[0]  for b in alive_filtered_bins]
    bin_centers_2 = [b[0]  for b in dead_filtered_bins]
    #print(len(bin_centers_1),len(alive_filtered_freqs))
    #print(len(bin_centers_2),len(dead_filtered_freqs))
    # Plot the histograms
    plt.figure(figsize=(12, 6))

    # Histogram 1
    plt.bar(
        bin_centers_1,
        alive_filtered_freqs,
        width=bin_width,
        color='blue',
        alpha=0.5,
        label='Alive',
        edgecolor='black',
        align='edge'
    )

    # Histogram 2
    plt.bar(
        bin_centers_2,
        dead_filtered_freqs,
        width=bin_width,
        color='green',
        alpha=0.5,
        label='Dead',
        edgecolor='black',
        align='edge'
    )
    
    # Add Separate Lines
    # Line for Alive
    plt.plot(
        [b + bin_width / 2 for b in bin_centers_1],  # X-coordinates (midpoints of bars)
        alive_filtered_freqs,  # Y-coordinates (frequencies)
        color='blue', linestyle='-', linewidth=2.0, label='Alive Line'
    )

    # Line for Dead
    plt.plot(
        [b + bin_width / 2 for b in bin_centers_2],  # X-coordinates (midpoints of bars)
        dead_filtered_freqs,  # Y-coordinates (frequencies)
        color='green', linestyle='-', linewidth=2.0, label='Dead Line'
    )
    

    # Add labels and legend
    plt.xlabel("Logarithmic Pdf Values", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title(f"Overlay of Two Histograms Lines with Non-Zero Bins Width={bin_width}", fontsize=14)
    plt.legend()
    #plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("Overlay of Two Histograms Lines with Non-Zero Bins")
    # Show the plot
    plt.show()
    
def make_collage():


    # Folder containing your saved images
    image_folder = r"D:\RA work Fall2024\grid_by_grid_dead_alive\dead_collage"
    output_file = "collage.png"

    # Get list of image files
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".png")]



    # Load all images and get their dimensions
    images = [Image.open(img) for img in image_files]
    img_width, img_height = images[0].size

    # Define grid size (6x5 for 26 images)
    columns, rows = 5, 5
    collage_width = columns * img_width
    collage_height = rows * img_height

    # Create blank canvas for the collage
    collage = Image.new("RGB", (collage_width, collage_height), (255, 255, 255))

    # Paste each image into the collage
    for idx, img in enumerate(images):
        x_offset = (idx % columns) * img_width
        y_offset = (idx // columns) * img_height
        collage.paste(img, (x_offset, y_offset))

    # Save the final collage
    collage.save(output_file)
    print(f"Collage saved as {output_file}")
