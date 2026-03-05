import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import hsv_to_rgb

import seaborn as sns 

import numpy as np 
import os

class analyze_dihedral:
    def __init__(self, data_path_list, link, title):
        self.data_path_list = data_path_list

        self.link = link

        self.title = title

        self.dframe_dihedral = pd.DataFrame()
        
        
    def generate_dframe_dihrdral(self):
        dfs = []
        # Loop through the files and read them into DataFrames
        for path in self.data_path_list:
            files = os.listdir(path)
            for file in files:
                filepath = os.path.join(path, file)
                column_name = file.split('.')[0]
                df = pd.read_csv(filepath, delimiter='\s+', engine='python')
                df.columns = ['step', column_name]
                # print(column_name)
                dfs.append(df)

        # Concatenate the second column of each DataFrame into a new DataFrame
        dframe_dihedral = pd.concat([df.iloc[:, 1] for df in dfs], axis=1)
        self.dframe_dihedral = dframe_dihedral

    def display_frequency_grid_box(self, hist2, title):
        try:
            # #print all the elements in hist2
            # for row in hist2:
            #     for element in row:
            #         print(element, end=" ")
            #     print()

            fig, ax = plt.subplots(figsize=(15,15))
            extent=[0,361,0,361]
            # norm = Normalize(vmin=np.min(hist2), vmax=np.max(hist2))
            # im = ax.imshow(hist2, extent=extent, cmap='YlGnBu', norm=norm, alpha=0.8)

            # # Create a colorbar with the same normalization
            # sm = plt.cm.ScalarMappable(cmap='YlGnBu', norm=norm)
            # sm.set_array([])  # Set an empty array to avoid data association
            # fig.colorbar(sm, ax=ax, label='Intensity')  # Add label to the colorbar

            # Annotate each cell with the numeric value
            num_rows, num_cols = hist2.shape
            for i in range(num_rows):
                for j in range(num_cols):
                    # Calculate the position of the text within the grid
                    x = (j + 0.5) * (extent[1] - extent[0]) / num_cols
                    y = (i + 0.5) * (extent[3] - extent[2]) / num_rows
                    value = hist2[i, j]  
                    text_color = 'white' if value == 0.00 else 'black'
                    ax.text(x, y, f'{value:.2f}',
                                    ha='center', va='center', color=text_color, fontsize=4)

            # Add grid lines 
            # ax.set_xticks(np.arange(0, num_cols, 1))
            # ax.set_yticks(np.arange(0, num_rows, 1))
            # ax.grid(which='both', color='w', linestyle='--', linewidth=0.5)

            ax.set_xlabel('Φ')
            ax.set_ylabel('Ψ')
            ax.set_title(title)
            ax.set_xticks(range(0,361,60))
            ax.set_yticks(range(0,361,60)) 
            plt.savefig(f'/home/rajan/Desktop/countour_plots_combined_traj/{self.title}_matrix.png', dpi=300)
            # ax.grid(True)

        except (TypeError, IndexError):
            print("Error creating subplots or accessing axes. Check your code.")
    
    def assignbins(self, dim, disc):
        minimum=float(dim[0])
        maximum=float(dim[1])
        bins =np.arange(minimum,(maximum+1),disc)
        return (bins) 
    
    def normalize_freq_matrix(self, hist2):
        total_sum = np.sum(hist2)
        normalized_matrix = hist2 / total_sum
        return(normalized_matrix * 100)
    
    def create_phipsi_bin_matrix(self):
        Xdim = (0, 361)
        Ydim = (0, 361)
        discX = 5
        discY = 5
        binsX= self.assignbins(Xdim, discX)
        binsY= self.assignbins(Ydim, discY)       

        hist2, edgesX, edgesY = np.histogram2d(self.dihedral_numpy_matrix[:,0], self.dihedral_numpy_matrix[:,1], bins = (binsX, binsY), weights=None)
        print(f'shape of the data matrix: {hist2.shape}')

        normal_matrix=  self.normalize_freq_matrix(hist2)

        # self.display_frequency_grid_box(normal_matrix.transpose(), self.title)
        
        return(normal_matrix, edgesX, edgesY)
    
    def generate_x_values_and_y_values(self):
        link = self.link

        # Then plot scatter points
        all_x1_values = []
        all_y1_values = []

        for lnk in link:
            angle_name1 = 'VPS_dihedral_phi-site' + str(lnk)
            angle_name2 = 'VPS_dihedral_psi-site' + str(lnk)

            x1 =[]
            for col in [str(angle_name1), ]:
                # print(col)
                for angle_val in self.dframe_dihedral[col]:
                    if angle_val < 0:
                        x1.append(angle_val+360)
                        all_x1_values.append(angle_val+360)
                    else:
                        x1.append(angle_val)
                        all_x1_values.append(angle_val)
                    
            y1 =[]
            for col in [str(angle_name2),]:
                for angle_val in self.dframe_dihedral[col]:                
                    if angle_val < 0:
                        y1.append(angle_val + 360)
                        all_y1_values.append(angle_val+360)
                    else:
                        y1.append(angle_val)
                        all_y1_values.append(angle_val)

        print('shape of final dots being used to generate a plot')
        print(len(all_x1_values),len(all_y1_values))

        self.all_x1_values = all_x1_values
        self.all_y1_values = all_y1_values
        #############################################
        dframe_dihedral_360 = pd.DataFrame({'phi':all_x1_values, 'psi':all_y1_values})
        self.dihedral_numpy_matrix = dframe_dihedral_360.to_numpy()

    def generate_color_bar(self):
        # Generate 50 colors with gradient effect
        colors = []
        for i in range(50):
            hue = i / 50  # This will give us the full spectrum
            saturation = min(1.0, (i + 1) / 25)  # Gradually increase saturation
            value = max(0.5, 1.0 - (i / 100))  # Gradually decrease brightness
            colors.append(hsv_to_rgb((hue, saturation, value)))
        
        self.colors =colors
  
    
    def generate_heatmap_population_density_plot4(self, x_label, y_label, title, colorbar_label):
        self.generate_x_values_and_y_values()
        heatmap, edgesX, edgesY = self.create_phipsi_bin_matrix()
        heatmap = heatmap.T

        fig, ((ax3, ax4), (ax1, ax2)) = plt.subplots(2, 2, figsize=(20, 20))

        x = np.arange(360/heatmap.shape[0]/2, 360, 360/heatmap.shape[0])
        y = np.arange(360/heatmap.shape[0]/2, 360, 360/heatmap.shape[0])
        X, Y = np.meshgrid(x, y)

        levels = 30        
        self.generate_color_bar()
        colors = self.colors
        n_bins = 50  # Number of color gradations
        cmap = LinearSegmentedColormap.from_list("white_to spectrum", colors, N=n_bins)
        cmap.set_bad('white')

        ###create a color bar to represent the 5 degree bin

        # Create masked array for zero values
        masked_heatmap = np.ma.masked_where(heatmap == 0.00, heatmap)

        # Create the filled contour plot
        contourf = ax1.contourf(X, Y, masked_heatmap, levels=levels, cmap=cmap)
        print(f'contourf: {contourf}')

        # Add a colorbar
        colorbar_label = 'Contour level'
        cbar = fig.colorbar(contourf, ax=ax4, )
        cbar.set_label(colorbar_label, rotation=90, labelpad=15, fontsize=20)
        cbar.ax.tick_params(labelsize=15) 

        ####################################        
        # Interpolate the data
        from scipy.ndimage import gaussian_filter
        heatmap_smooth = gaussian_filter(masked_heatmap, sigma=0.5)
        ####################################         
        masked_heatmap = np.ma.masked_where(heatmap_smooth == 0.00, heatmap_smooth) 

        # Create the filled contour plot
        contourf = ax1.contourf(X, Y, masked_heatmap, levels=levels, cmap=cmap)
        print(f'contourf: {contourf}')          

        # Create the contour lines
        contour = ax1.contour(X, Y, heatmap_smooth, levels=levels, colors='black', linewidths=0.4)

        print(f'contour: {contour.levels} \n and length = {len(contour.levels)}')
        
        # Calculate sum between contour levels
        def calculate_region_sum(data_matrix, level_lower, level_upper):
            #only find the sum of values in the first 48x72 submatrix
            # data = data_matrix[:48, :]
            data = data_matrix

            mask = (data >= level_lower) & (data < level_upper)
            return np.sum(data[mask])

        # Generate labels with sums for each contour level
        fmt = {}
        for i in range(len(contour.levels)-1):
            level_lower = contour.levels[i]
            level_upper = contour.levels[i+1]
            region_sum = calculate_region_sum(masked_heatmap, level_lower, level_upper)
            # Format the sum as percentage with one decimal place
            fmt[contour.levels[i]] = f'{region_sum:.1f}'
        
        # Add the last level
        fmt[contour.levels[-1]] = f'{calculate_region_sum(masked_heatmap, contour.levels[-1], np.inf):.1f}'

        #####################################
        print('\n The sum of the fmt:')
        print(fmt)
        print(len(fmt.values()))
        print(fmt.values())
        float_list = [float(x) for x in fmt.values() if x != '--']
        print(sum(float_list))
        ####################################

        # Label the contour lines with the calculated sums
        clabels = ax4.clabel(contour, inline=True, fontsize=3.0, fmt=fmt)
        # Iterate through the text objects and set their color
        for txt in clabels:
            # txt.set_color('red')
            txt.set_weight('bold')  

        # # Set x and y axis ticks and labels
        # ax4.set_xticks(np.arange(0, 361, 30))
        # ax4.set_yticks(np.arange(0, 361, 30))        

        ##############################################       
        # ax1.scatter(self.all_x1_values, self.all_y1_values,  color ='red', marker='.', s=1)

        ##############################################
        # Plot the histogram of the frequency on the right subplot (ax2)
        sns.histplot(y=self.all_y1_values, ax=ax2, binwidth=5, color ='blue', kde=True)
        # sns.histplot(y=all_y2_values, ax=ax2, bins=20, color ='orange', kde=True)
        ax2.set_xlabel('Frequency', fontsize =20)
        # ax2.set_ylabel('Ψ', fontsize =20)
        # ax2.set_title('')
        ax2.set_xlim(xmin = 0, xmax = 40000)
        ax2.set_ylim(ymin = 0, ymax = 361)
        # ax2.legend(loc = 'upper left', fontsize = 20)
        ax2.tick_params(axis='both', which='both', labelsize=15)  # Set tick font size
        ax2.set_yticks(np.arange(0, 360, 30))
        ax2.tick_params(axis='y', which='both', length=0)
        ax2.tick_params(axis='both', which='both', labelleft =False)
        ax2.xaxis.set_major_locator(ticker.MultipleLocator(base=8000))  # Set x-ticks at a distance of 50,000
            
        #####################
        # Plot the histogram of the frequency on the top subplot (ax3)
        sns.histplot(x=self.all_x1_values, ax=ax3, binwidth=5, color ='blue', kde=True)
        # sns.histplot(x=all_x2_values, ax=ax3, bins=20, color ='orange', kde=True)
        ax3.set_ylabel('Frequency', fontsize =20)
        # ax3.set_xlabel('Ψ', fontsize =20)
        # ax3.set_title('')
        ax3.set_ylim(ymin = 0, ymax = 40000)
        ax3.set_xlim(xmin = 0, xmax = 361)
        # ax3.legend(loc = 'upper left', fontsize = 20)
        ax3.tick_params(axis='both', which='both', labelsize=15)  # Set tick font size
        ax3.set_xticks(np.arange(0, 360, 30))
        ax3.tick_params(axis='x', which='both', length=0)
        ax3.tick_params(axis='both', which='both', labelbottom = False)  # Set tick font size
        ax3.yaxis.set_major_locator(ticker.MultipleLocator(base=8000))  # Set x-ticks at a distance of 50,000

        # Hide ax4 
        ax4.axis('off')    

        #set legend labels 
        # ax1.scatter([], [], marker= '.', s=0.5, color='blue', label='')
        # ax1.scatter([], [], marker= '.', s=0.5, color='orange', label='Rhap-(1-4)-α-GalpA')

        #set limit for x and y ticks 
        ax1.set_xlim(xmin=0, xmax = 361)
        ax1.set_ylim(ymin=0, ymax= 361)

        ax1.set_xticks(np.arange(0, 360, 30))
        ax1.set_yticks(np.arange(0, 360, 30))
        ax1.tick_params(axis='both', which='both', labelsize=15)  # Set tick font size
     
        # set the x and y axis labels
        ax1.set_xlabel('ϕ', fontsize=20)
        ax1.set_ylabel('Ψ', fontsize=20)

        # add a title to the plot
        # ax1.set_title('Fuc-α(1-4)-GlcA and Rha-α(1-4)-GalA, TIP5P', fontsize=20)
        
        # show the legend
        # ax1.legend(loc='upper right', bbox_to_anchor=(0.5, 1), fontsize=20)

        # increase the font size of the X and Y tick labels
        # plt.xticks(fontsize=18)
        # plt.yticks(fontsize=18)
        
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        fig.suptitle(self.title, fontsize=20)

        # Show grid lines on the plots
        ax1.grid(False)
        ax2.grid(True)
        ax3.grid(True)
        ax4.grid(False)

        # plt.tight_layout()
        plt.savefig(f'./{self.title}_color4.png', format ='png', dpi=550)
        # show the plot
        plt.show()   

