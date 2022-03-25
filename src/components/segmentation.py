import numpy as np
import random
import os
from numba import jit
import math


@jit(nopython=True,cache=True)
def map_pixels(image_flat,cluster_centers,pixel_mapping_temp,num_cluster_centers,num_image_rows):
    """
    Called from Segmentation.k_means_segmentation. Maps all image pixels to a cluster by taking a 1x3 pixel and
    calculating the distance between every cluster center. Appends pixels location to the cluster it has minimum
    distance to.

    Parameters:
    -----------

        image_flat (numpy array) : the 3D flattened image to calculate the distances over
        cluster_centers (numpy array) : array of all cluster centroids
        pixel_mapping_temp (numpy array) : the mapping of pixel to centroid. This array is passed and changed by reference.
        num_cluster_centers (int) : the number of clusters
        num_image_rows (int) : the length of image_flat which is equal to the number of pixels in the image

    Returns:
    --------
        
        None
    """

    # For every pixel in the image
    for i in range(num_image_rows):
        distances = []
        # For every cluster calculate which one the pixel is closest to
        for j in range(num_cluster_centers):
            # calculating Euclidean distance for the pixel against each cluster
            distances.append(np.linalg.norm(image_flat[i][:] - cluster_centers[j]))
        # For every pixel map to which cluster it belongs based on which one it is closest to
        pixel_mapping_temp[i] = distances.index(min(distances))


@jit(nopython=True,cache=True)
def recalculate_cluster_centers(image_flat,cluster_centers,pixel_to_cluster_mapping,num_cluster_centers,num_image_rows):
    """
    Called from Segmentation.k_means_segmentation.Recalculate cluster centers by averaging of all pixel values that
    are assigned to the cluster. This cluster to pixel assignment is contained in the pixel_to_cluster_mapping array.
    Cluster centers are then moved to their new centers of gravity. cluster_centers is passed and changed by reference.

    Parameters:
    -----------

        image_flat (numpy array) : the 3D flattened image
        cluster_centers (numpy array) : array of all cluster centroids. Passed and changed by reference.
        pixel_to_cluster_mapping (numpy array) : the mapping of pixel to centroid.
        num_cluster_centers (int) : the number of clusters
        num_image_rows (int) : the length of image_flat which is equal to the number of pixels in the image

    Returns:
    --------
         
        None
    """      
    
    # Calculate every new cluster center
    for i in range(num_cluster_centers):
        cluster_sum = np.zeros(3)
        cluster_average = np.zeros(3)
        pixel_count = 0

        for j in range(num_image_rows):
            # If a pixel belongs to a cluster then sum its values with other cluster pixels
            if pixel_to_cluster_mapping[j] == i:
                cluster_sum = cluster_sum + image_flat[j][:]
                pixel_count = pixel_count + 1

        # Only recalculate cluster center if a pixel is assigned to it (otherwise center will move to [0,0,0])
        if pixel_count != 0:
            cluster_average = cluster_sum/pixel_count # get average of pixel values
            cluster_centers[i][:] = cluster_average # move cluster center


@jit(nopython=True,cache=True)
def segment_image(image_flat,cluster_centers,pixel_to_cluster_mapping,num_cluster_centers,num_image_rows):
    """
    Called from Segmentation.k_means_segmentation. Perform pixel assignments to appropriate cluster once 
    all pixels are assigned.

    Parameters:
    -----------

        image_flat (numpy array) : the 3D flattened image
        cluster_centers (numpy array) : array of all cluster centroids
        pixel_to_cluster_mapping (numpy array) : the mapping of pixel to centroid.
        num_cluster_centers (int) : the number of clusters
        num_image_rows (int) : the length of image_flat which is equal to the number of pixels in the image

    Returns:
    --------

        None
    """

    # Calculate every new cluster center
    for i in range(num_cluster_centers):            
        for j in range(num_image_rows):
            # If a pixel belongs to a cluster reassign it's value to that of it's cluster centroid
            if pixel_to_cluster_mapping[j] == i:
                image_flat[j][:] = cluster_centers[i][:]


class Segmentation:
    """
    Component class for segmentation techniques. Uses some functions outside of this class which can be just in time
    compiled by Numba. This provides a huge speed boost for these numpy based operations.

    Attributes
    ----------

    num_cluster_centers(int) : The number of cluster centers to use for K-means segmentation. Loaded from .env file.
    
    Methods
    -------

    k_means_segmentation(self,image)
        K-means segmentation algorithm for an RGB image. Implemented as a wrapper for Numba compiled non-class methods.

    histogram_thresholding_segmentation(self,image,bin_values,bins)
        Performs segmentation through histogram thresholding of a grayscale image.
    """

    def __init__(self):
        self.num_cluster_centers = int(os.getenv('NUM_K_MEANS_CLUSTERS'))
    

    def k_means_segmentation(self,image):
        """
        K-means segmentation algorithm for an RGB image.

        Parameters:
        -----------

            image (numpy array) : 3D rgb image

        Returns:
        --------

            image (numpy array) : the segmented image as a 3D numpy array
        """

        image_copy = np.copy(image) # make copy of image

        # reshape the array into 2D matrix of 3 columns with each column representing a color spectrum
        image_flat = np.reshape(image_copy, (image_copy.shape[0] * image_copy.shape[1], image_copy.shape[2]))
        num_image_rows, num_image_columns = image_flat.shape

        # initialize new array with height equal to number of clusters and width of num_image_columns
        cluster_centers = np.empty([self.num_cluster_centers,num_image_columns],dtype=np.float64)

        # list[i] of which pixel of image_flat[i][:] belongs to which cluster
        pixel_to_cluster_mapping = np.empty([num_image_rows,1],dtype=np.float64)
        pixel_mapping_temp = np.copy(pixel_to_cluster_mapping) # temp mapping array

        # randomly initialize cluster centers
        for i in range(self.num_cluster_centers):
            cluster_centers[i][0] = random.randint(0, 255)  # red
            cluster_centers[i][1] = random.randint(0, 255)  # green
            cluster_centers[i][2] = random.randint(0, 255)  # blue
        
        pixel_changed_cluster_flag = True

        # Perform K-means segmentation. Loop as long as pixels are being reassigned to clusters.
        while pixel_changed_cluster_flag:
            # pixel_mapping_temp numpy array passed by reference 
            map_pixels(image_flat,cluster_centers,pixel_mapping_temp,self.num_cluster_centers,num_image_rows)

            # Check if any pixels changed clusters
            comparison = pixel_to_cluster_mapping == pixel_mapping_temp        
            if comparison.all():
                pixel_changed_cluster_flag = False
            else:
                pixel_changed_cluster_flag = True # signal that pixels changed cluster assignments
                pixel_to_cluster_mapping = np.copy(pixel_mapping_temp) # set mapping to temp mapping

                # calculate new cluster_centers based on average of pixels assigned to it
                recalculate_cluster_centers(image_flat,cluster_centers,pixel_to_cluster_mapping,self.num_cluster_centers,num_image_rows)
                print(cluster_centers)

        # Reassign image pixel values to the cluster centroid values
        segment_image(image_flat,cluster_centers,pixel_to_cluster_mapping,self.num_cluster_centers,num_image_rows)
        return np.reshape(np.rint(image_flat),image.shape)


    def histogram_thresholding_segmentation(self,image,bin_values,bins):
        """
        Performs segmentation through histogram thresholding of a 2D grayscale image.

        Parameters:
        -----------

            image (numpy 2D array) : greyscale image to be segmented
            bin_values(numpy array): represents the number of pixels in each histogram bin
            bins (numpy array): defines the bins which should be matched with histogram pixel values

        Returns:
        --------

            segmented_image (numpy 2D array) : the segmented (0 or 255) 2D image
        """
        num_pixels = image.size # of pixels in the image
        within_group_variance = [] # within group variance list
        # From 0 to 255
        for threshold in range(len(bins)-1):
            p_o = 0
            p_b = 0
            mean_o = 0
            mean_b = 0
            variance_o = 0
            variance_b = 0
            # Calculate mean and variance for o and b at T
            # From 0 to Threshold
            for j in range(threshold+1):
                P_j = bin_values[j]/num_pixels
                p_o = p_o + P_j
                # Avoid divide by zero errors
                if p_o == 0:
                    mean_o = mean_o + 0
                    variance_o = variance_o + 0
                else:
                    mean_o = mean_o + ((j*P_j)/p_o)
                    variance_o = variance_o + (((j-mean_o)**2)*P_j/p_o)

                if math.isnan(variance_o):
                    variance_o = 0

            # From Threshold+1 to 255
            for k in range(threshold+1,len(bins)-1):
                P_k = bin_values[k]/num_pixels
                p_b = p_b + P_k
                if p_b == 0:
                    mean_b = mean_b + 0
                    variance_b = variance_b + 0
                else:
                    mean_b = mean_b + ((k*P_k)/p_b)
                    variance_b = variance_b + (((k-mean_b)**2)*P_k/p_b)
                
                if math.isnan(variance_b):
                    variance_b = 0
        
            group_variance = variance_o*p_o + variance_b*p_b
            within_group_variance.append(group_variance)

        # Get the threshold where segmentation will occur
        optimum_threshold = within_group_variance.index(min(within_group_variance))

        # Flatten the image for segmentation
        image_list = list(image.flatten())
        segmented_image = []

        # Segment around the threshold
        for pixel_intensity in image_list:
            if pixel_intensity <= optimum_threshold:
                segmented_image.append(0)
            else:
                segmented_image.append(255)

        # Reshape image into 2D and return
        return np.reshape(np.asarray(segmented_image), image.shape)
