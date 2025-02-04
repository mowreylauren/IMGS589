#Lauren Mowrey
#IMGS 589
#Homework 1
#1/23/25
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from matplotlib.colors import LogNorm
import pandas as pd


#Problem 1
#loading data
data = np.load('sentinel2_rochester.npy')

#creating photogrid
rows, cols = 3,4
fig, axes = plt.subplots(rows, cols, figsize=[15,10])

#listing band labels
band_labels = [
    "1 - Coastal Aerosol (443nm)",
    "2 - Blue (490nm)",
    "3 - Green (560nm)",
    "4 - Red (665nm)",
    "5 - Red Edge 1 (705nm)",
    "6 - Red Edge 2 (740nm)",
    "7 - Red Edge 3 (783nm)",
    "8 - NIR (842nm)",
    "8A - Narrow NIR (865nm)",
    "9 - Water Vapor (940nm)",
    "11 - SWIR 1 (1610nm)",
    "12 - SWIR 2 (2190nm)"
]
#clipping anywhere data is equal to zero (no data)
data_clipped = np.where(data == 0, np.nan, data)

#displaying each band
for i in range(rows*cols):
    grid = axes[i//cols, i%cols]
    im = grid.imshow(data_clipped[:, :, i], cmap='terrain')
    grid.set_axis_off()
    grid.set_title(band_labels[i])

#adding a colorbar
fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
plt.show()

#Problem 2
#2a. calculating statistics
def calculate_band_statistics(args):
    flat_args = args.flatten()
    stats = {
    'mean' : np.mean(flat_args),               #average of all values
    'std' : np.std(flat_args),                 #variation from the mean
    'q1' : np.percentile(flat_args, 25),    #25th percentile
    'q3' : np.percentile(flat_args, 75),    #75th percentile
    'min' : flat_args.min(),                   #minimum value
    'max' : flat_args.max(),                   #maximum value
    'skewness' : skew(flat_args),              #deviaiton from symmetric distribution
    'kurtosis' : kurtosis(flat_args)}          #how peaked or flat distribution is compared to normal
    return stats

#printing results
for i in range(12):
    stats = calculate_band_statistics(data[:, :, i])
    print(f'Band {band_labels[i]}:\n'
          f" Mean = {stats['mean']}\n"
          f" Standard Deviation = {stats['std']}\n"
          f" Minimum = {stats['min']}\n"
          f" Maximum = {stats['max']}\n"
          f" 1st Quartile = {stats['q1']}\n"
          f" 3rd Quartile = {stats['q3']}\n"
          f" Skewness = {stats['skewness']}\n"
          f" Kurtosis = {stats['kurtosis']}\n"
          )

#2b. calculating z score
def standardize(arr):
    arr_std = (arr - np.mean(arr))/np.std(arr)
    return arr_std

#creating histograms
fig, axes = plt.subplots(rows, cols, figsize=[15,10])
axes = axes.flatten()
for i in range(rows*cols):
    flat_data = data[:,:,i].flatten()
    data_standard = standardize(flat_data)

    #creating a mask for outliers
    mask1 = data_standard > 3
    mask2 = data_standard < -3
    mask = mask1 | mask2
    outliers = data_standard[mask]
    inliers = data_standard[~mask]

    #ploting histogram and highlighting outliers
    axes[i].hist(inliers, bins=100, color='blue', alpha=0.7, label="Inliers")
    axes[i].hist(outliers, bins=20, color='red', alpha=0.7, label="Outliers")
    outlier_sum = mask.sum()
    axes[i].set_title(f'{band_labels[i]}\n number of outliers: {outlier_sum}')
    axes[i].legend()
fig.tight_layout(pad=5.0)
plt.show()

#Problem 3
#3a. correlation matrix
def  correlation_matrix(args):
    flattened_data = args.reshape(-1, 12)
    #correlation matrix for the 12 bands
    corr_matrix = np.corrcoef(flattened_data, rowvar=False)
    return corr_matrix

#computing correlation matrix
corr_matrix = correlation_matrix(data)
print(f'correlation matrix (12 band):\n{corr_matrix}')

#plotting correlation matrix
band_numbers = ['1','2','3','4','5','6','7','8','8A','9','11','12']
plt.title("Correlation Matrix")
mat = plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest', vmin=-1, vmax=1)
plt.xticks(list(range(0,12)),band_numbers) #horizontal tick mark
plt.yticks(list(range(0,12)),band_numbers) #vertical tick mark
plt.colorbar(mat, label="Correlation Coefficient") #adding colorbar
plt.show()

#3b. pairwise scatter and density plots
def correlation_plot(data, bands, bins=100):
    #reshape to specified bands
    flat_data = data.reshape(-1, data.shape[-1])
    selected_bands = flat_data[:, bands]

    #calculating correlation matrix
    corr_matrix = np.corrcoef(selected_bands, rowvar=False)

    #creating subplots
    num_bands = len(bands)
    fig, axes = plt.subplots(num_bands, num_bands, figsize=(12, 12), constrained_layout=True)
    for i in range(num_bands):
        for j in range(num_bands):
            ax = axes[i, j]

            if i == j:  #diagonal 1D histogram
                data_band = selected_bands[:, i]
                ax.hist(data_band, bins=bins, alpha=0.7)
                ax.set_title(f"Density (Band {bands[i] + 1})")

            else:  #off-diagonal scatter plot with density shading using 2D histogram
                x, y = selected_bands[:, j], selected_bands[:, i]
                hist, xedges, yedges = np.histogram2d(x, y, bins=bins)

                #log normalization for better visualization
                norm = LogNorm(vmin=1, vmax=hist.max())

                #plotting 2D histogram
                ax.imshow(hist.T, extent=[x.min(), x.max(), y.min(), y.max()], origin="lower",
                          cmap="viridis", aspect="auto", norm=norm, interpolation="nearest")

                ax.set_xlabel(f"Band {bands[j] + 1}")
                ax.set_ylabel(f"Band {bands[i] + 1}")

    #plotting combined
    plt.suptitle("Pairwise Scatter and Density Plots", fontsize=20)
    plt.show()

    return corr_matrix

#specify the bands
bands = [1, 2, 3, 7]

#compute correlation matrix
corr_matrix = correlation_plot(data, bands, bins=100)

#display correlation matrix
print(f'Correlation Matrix (4 band): \n {corr_matrix}')

#Problem 4
#loading oak and road data
oak_data = pd.read_csv('oak.txt', sep='\t' ,names=['Wavelength', 'Reflectance'])
road_data = pd.read_csv('road.txt', sep='\t' ,names=['Wavelength', 'Reflectance'])

#naming variables, dividing reflectance by 100
wavelengths_oak = oak_data['Wavelength']
reflectance_oak = oak_data['Reflectance'] / 100
wavelengths_road = road_data['Wavelength']
reflectance_road = road_data['Reflectance'] / 100

#interpolation for spectral downsampling
sentinel_bands = [0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 1.610, 2.190]
road_downsample = np.interp(sentinel_bands, wavelengths_road, reflectance_road)
oak_downsample = np.interp(sentinel_bands, wavelengths_oak, reflectance_oak)

#computing cosine similarity
def sam(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    angle = np.arccos(dot_product / norm_product)
    return angle

#flattening data
used_bands = data[:,:,[1,2,3,4,5,6,7,8,10,11]]
data_reshaped = used_bands.reshape(-1, used_bands.shape[-1])

#calculating spectral angles
angles_oak = np.array([sam(pixel, oak_downsample) for pixel in data_reshaped])
angles_road = np.array([sam(pixel, road_downsample) for pixel in data_reshaped])

#identifying 100 closest matches
closest_pixels_oak = np.argsort(angles_oak)[:100]
closest_pixels_road = np.argsort(angles_road)[:100]

#converting 1D indices back to 2D coordinates
pixels_oak = np.unravel_index(closest_pixels_oak, (954, 716))
pixels_road = np.unravel_index(closest_pixels_road, (954, 716))

#plotting the 1st, 50th, and 100th closest matches for oak data
plt.figure(figsize=(10, 5))
for i, idx in enumerate([0, 49, 99]):
    plt.plot(sentinel_bands, data_reshaped[closest_pixels_oak[idx]], label=f'Match {idx+1}')
plt.plot(sentinel_bands, oak_downsample, 'k--', label='ECOSTRESS Oak', linewidth=2)
plt.xlabel('Wavelength (µm)')
plt.ylabel('Reflectance')
plt.legend()
plt.title('Comparison of Closest Sentinel-2 Matches with ECOSTRESS Oak')
plt.show()

#plot the 1st, 50th, and 100th matches for road data
plt.figure(figsize=(10, 5))
for i, idx in enumerate([0, 49, 99]):
    plt.plot(sentinel_bands, data_reshaped[closest_pixels_road[idx]], label=f'Match {idx+1}')
plt.plot(sentinel_bands, road_downsample, 'k--', label='ECOSTRESS Road', linewidth=2)
plt.xlabel('Wavelength (µm)')
plt.ylabel('Reflectance')
plt.legend()
plt.title('Comparison of Closest Sentinel-2 Matches with ECOSTRESS Road')
plt.show()

#setting a cutoff angle threshold
threshold= 0.4
classified_oak = angles_oak.reshape(954, 716) < threshold
classified_road = angles_road.reshape(954, 716) < threshold

#displaying pixels where oak is present
plt.figure()
plt.imshow(classified_oak, cmap="gray")
plt.title("Identified Oak Pixels (white)")
plt.axis('off')
plt.show()

#displaying pixels where road is present
plt.figure()
plt.imshow(classified_road, cmap="gray")
plt.title("Identified Road Pixels (white)")
plt.axis('off')
plt.show()
