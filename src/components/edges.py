import numpy as np
from numba import jit


class Edges:
    """
    Component class for edge techniques. 

    Attributes
    ----------

    
    
    Methods
    -------

    
    """

    def __init__(self):
        pass


    @staticmethod
    @jit(nopython=True,cache=True)
    def sobel_edge_detection(image,threshold = 2.0):
        """
        Perform Sobel edge detection algorithm using improved sobel filter.
        """

        # x direction gradient filter
        Hx = np.array([[-3,0,3],
                       [-10,0,10],
                       [-3,0,3]])

        # y direction gradient filter
        Hy = np.array([[-3,-10,-3],
                       [0,0,0],
                       [3,10,3]])
              
        sobel_division_factor = 32 # division factor for sobel edge detection

        threshold = threshold # threshold for gradient magnitude to be marked as an edge   

        # Getting the dimensions of the image
        height, width = image.shape
        x_image_copy= np.copy(image) # make copy of image size-1 for x gradient
        y_image_copy= np.copy(image) # make copy of image size-1 for y gradient     

        # Compute the x and y gradients
        # Pixels on outer edge of image (those which cause the filter to be off the image) will be ignored
        # The outer two for loops move the filter central pixel over the image
        for row in range(1,height-1):
            for column in range(1,width-1):
                sum_x = 0
                sum_y = 0
                # These two for loops do the averaging within the filters bounds
                for j in range(-1, 2): # moves across the columns of the filter
                    for i in range(-1, 2): # moves down the rows of the filter
                        # multiply pixel by its weighting factor from the x filter
                        pixel_x = image[row+i][column+j] * Hx[i+1][j+1]
                        # multiply pixel by its weighting factor from the y filter
                        pixel_y = image[row+i][column+j] * Hy[i+1][j+1]
                        sum_x = sum_x + pixel_x # sum all pixels within the neighborhood for x gradient
                        sum_y = sum_y + pixel_y # sum all pixels within the neighborhood for y gradient

                # divide by the division factor and store into x image copy
                x_image_copy[row][column] = round(sum_x/sobel_division_factor)
                # divide by the division factor and store into y image copy
                y_image_copy[row][column] = round(sum_y/sobel_division_factor)

        # Calculate gradient magnitude from x and y derivatives
        gradient_magnitude = np.sqrt(np.square(x_image_copy) + np.square(y_image_copy))

        # Matrix for the edges
        gradient_edges = np.copy(gradient_magnitude)

        

        # Mark edges if its magnitude is a maxima in a given direction (3x3 area)
        # Start at 2 because outer edge of pixels was never run over by the derivative filter
        for row in range(2,height-2):
            for column in range(2,width-2):
                hot_point = gradient_magnitude[row][column] # pixel being evaluated
                # check x direction maxima
                if max(gradient_magnitude[row-1][column],hot_point,gradient_magnitude[row+1][column]) == hot_point:
                    if hot_point > threshold:
                        gradient_edges[row][column] = 0 # mark as an edge
                    else:
                        gradient_edges[row][column] = 255 # mark as not an edge                    
                # check y direction maxima
                elif max(gradient_magnitude[row][column-1],hot_point,gradient_magnitude[row][column+1]) == hot_point:
                    if hot_point > threshold:
                        gradient_edges[row][column] = 0 # mark as an edge
                    else:
                        gradient_edges[row][column] = 255 # mark as not an edge
                else:
                    gradient_edges[row][column] = 255 # mark as not an edge
                    
        # return gradient edges with two out layers sliced out
        return gradient_edges[2:height-2,2:width-2]
        