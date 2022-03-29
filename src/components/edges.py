import numpy as np
from numba import jit
import math


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
                # Multiply kernel times image patch the hot point is over and sum the elements
                x_image_copy[row][column] = np.sum(Hx*image[row-1:row+2,column-1:column+2])
                # divide by the division factor and store into y image copy
                y_image_copy[row][column] = np.sum(Hy*image[row-1:row+2,column-1:column+2])

        # divide by the division factor
        x_image_copy = x_image_copy/sobel_division_factor
        # divide by the division factor
        y_image_copy = y_image_copy/sobel_division_factor

        # Calculate gradient magnitude from x and y derivatives
        gradient_magnitude = np.sqrt(np.square(x_image_copy) + np.square(y_image_copy))

        # Matrix for storing the edges
        gradient_edges = np.copy(gradient_magnitude)        

        # Mark edges if its magnitude is a maxima in a given direction (3x3 area)
        # Start at 2 because outer edge of pixels was never run over by the derivative filter
        for row in range(2,height-2):
            for column in range(2,width-2):
                hot_point = gradient_magnitude[row][column] # pixel being evaluated
                # check x direction maxima
                if max(gradient_magnitude[row-1][column],hot_point,gradient_magnitude[row+1][column]) == hot_point:
                    # if maxima is greater than threshold then mark as edge
                    if hot_point > threshold:
                        gradient_edges[row][column] = 0 # mark as an edge
                    else:
                        gradient_edges[row][column] = 255 # mark as not an edge                    
                # check y direction maxima
                elif max(gradient_magnitude[row][column-1],hot_point,gradient_magnitude[row][column+1]) == hot_point:
                    # if maxima is greater than threshold then mark as edge
                    if hot_point > threshold:
                        gradient_edges[row][column] = 0 # mark as an edge
                    else:
                        gradient_edges[row][column] = 255 # mark as not an edge
                else:
                    gradient_edges[row][column] = 255 # mark as not an edge
                    
        # return gradient edges with two outer layers sliced out
        return gradient_edges[2:height-2,2:width-2]
        

    @staticmethod
    @jit(nopython=True,cache=True)
    def edge_dilation(image_edges, num_layers=1):
        """
        Perform dilation of an edge (black and white, eg 0 or 255) based image.
        """

        # Quasi-Disk shaped structuring elements                                     
        _structuring_element = np.array([[False,True,True,True,False],
                                        [True,True,True,True,True],
                                        [True,True,True,True,True],
                                        [True,True,True,True,True],
                                        [False,True,True,True,False]])

        structuring_element = np.array([[True,True,True],
                                        [True,True,True],
                                        [True,True,True]])

        __structuring_element = np.array([[False,True,False],
                                        [True,True,True],
                                        [False,True,False]])

        structure_height, structure_width = structuring_element.shape
        floor_structure_height = math.floor(structure_height/2)
        floor_structure_width = math.floor(structure_width/2)

        # get size of image
        height, width = image_edges.shape
        # copy image_edges
        image_edges_copy = np.copy(image_edges)
        edge_dilation_temp = np.copy(image_edges)  

        # Add number of specified layers
        for _ in range(num_layers):
            # go through every pixel of image
            # replace black pixel (intensity == 0) with the structuring element
            for row in range(floor_structure_height,height-floor_structure_height):
                for column in range(floor_structure_width,width-floor_structure_width):
                    # if over an edge pixel apply the structuring element to it (intensity 0 == black)
                    if image_edges_copy[row][column] == 0:
                        # apply structuring element for edge dilation
                        # numpy.where() iterates over the structuring element bool array
                        # and for every True it yields corresponding element from the first list (0 == for a black edge pixel)
                        # and for every False it yields corresponding element from the second list
                        edge_dilation_temp[row-floor_structure_height:row+floor_structure_height+1,column-floor_structure_width:column+floor_structure_width+1] = np.where(structuring_element,
                                                                                                                                                                            np.zeros((structure_height,structure_width)),
                                                                                                                                                                            image_edges_copy[row-floor_structure_height:row+floor_structure_height+1,column-floor_structure_width:column+floor_structure_width+1])
                        
            image_edges_copy = np.copy(edge_dilation_temp) # copy over in preparation for adding another layer

        return image_edges_copy


    @staticmethod
    @jit(nopython=True,cache=True)
    def edge_erosion(image_edges, num_layers=1):
        """
        Perform edge erosion of an edge based image.
        """

        structuring_element = np.array([[False,True,False],
                                        [True,True,True],
                                        [False,True,False]])

        structure_height, structure_width = structuring_element.shape
        floor_structure_height = math.floor(structure_height/2)
        floor_structure_width = math.floor(structure_width/2)

        # get size of image
        height, width = image_edges.shape
        # copy image_edges
        image_edges_copy = np.copy(image_edges)
        edge_erosion_temp = np.copy(image_edges)

        # Remove number of specified layers
        for _ in range(num_layers):
            # go through every pixel of image
            for row in range(floor_structure_height,height-floor_structure_height):
                for column in range(floor_structure_width,width-floor_structure_width):
                    # if over an edge pixel apply the structuring element to it (intensity 0 == black)
                    if image_edges_copy[row][column] == 0:
                        # mapping of structuring element with zeros array and image
                        # numpy.where() iterates over the structuring element bool array
                        # and for every True it yields corresponding element from the first list (0 == for a black edge pixel)
                        # and for every False it yields corresponding element from the second list (image_edges_copy pixel)
                        mapping = np.where(structuring_element,
                                            np.zeros((structure_height,structure_width)),
                                            image_edges_copy[row-floor_structure_height:row+floor_structure_height+1,column-floor_structure_width:column+floor_structure_width+1])

                        # if structuring element present in edge image then retain pixel
                        if np.array_equal(mapping,image_edges_copy[row-floor_structure_height:row+floor_structure_height+1,column-floor_structure_width:column+floor_structure_width+1]):
                            edge_erosion_temp[row][column] = 0
                        # else remove pixel
                        else:
                            edge_erosion_temp[row][column] = 255

            image_edges_copy = np.copy(edge_erosion_temp) # copy over in preparation for removing another layer

        return image_edges_copy
