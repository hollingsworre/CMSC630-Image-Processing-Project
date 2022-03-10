import math
import numpy as np


class ImagePointOperations:
    """
    Component class for all image point operations available to work with.

    Methods:
    --------

    smooth2dImage(image, weighted_filter)
        2D Image smoothing with a user specified (via .env file) averaging filter (Box or Gaussian)

    difference2dImage(image,weighted_filter)
        2D Image difference with a user specified (via .env file) Laplacian filter

    medianOf2dImage(image,weighted_filter)
        2D Image median with a user specified (via .env file) median filter
    """

    def __init__(self):
        pass

    def smooth2dImage(self,image,weighted_filter):
        """
        2D Image smoothing with a user specified (via .env file) averaging filter (Box or Gaussian)
        
        Parameters:
        -----------
            image(numpy array): the image to be smoothed

            weighted_filter: the weighted filter used to smooth the image::

                dict:{
                    'filter': numpy matrix,
                    'column_vector': None(if not separable) or numpy array(if separable),
                    'row_vector': None(if not separable) or numpy array(if separable)
                }

        Returns:
        --------
            image(numpy array): the smoothed image
        """

        # Getting the dimensions of the image
        height, width = image.shape
        image_copy= np.copy(image) # make copy of image

        filter_height, filter_width = weighted_filter.shape # get width and height of filter
        floor_filter_height = math.floor(filter_height/2)
        floor_filter_width = math.floor(filter_width/2)

        # Pixels on outer edge of image (those which cause the filter to be off the image) will be ignored
        # The outer two for loops move the filter central pixel over the image
        for row in range(floor_filter_height,height-floor_filter_height):
            for column in range(floor_filter_width,width-floor_filter_width):
                sum = 0
                # These two for loops do the averaging within the filters bounds
                for j in range(-1*floor_filter_width, floor_filter_width+1): # moves across the columns of the filter
                    for i in range(-1*floor_filter_height, floor_filter_height+1): # moves down the rows of the filter
                        # multiply pixel by its weighting factor from the filter
                        p = image[row+i][column+j] * weighted_filter[i+floor_filter_height][j+floor_filter_width] 
                        sum = sum + p # sum all pixels within the neighborhood

                # divide by the number of pixels and store into image copy
                image_copy[row][column] = round(sum/weighted_filter.sum())

        return image_copy # return averaged image


    def difference2dImage(self,image,weighted_filter):
        """
        2D Image difference with a user specified (via .env file) Laplacian filter
        
        Parameters:
        -----------
            image(numpy array): the image to be smoothed

            weighted_filter : the Laplacian filter used to find differenece within the image::

                dict:{
                        'filter': numpy matrix,
                        'column_vector': None(if not separable) or numpy array(if separable),
                        'row_vector': None(if not separable) or numpy array(if separable)
                    }

        Returns:
        --------
            image(numpy array): the altered image
        """

        # Getting the dimensions of the image
        height, width = image.shape
        image_copy= np.copy(image) # make copy of image

        filter_height, filter_width = weighted_filter.shape # get width and height of filter
        floor_filter_height = math.floor(filter_height/2)
        floor_filter_width = math.floor(filter_width/2)

        # Pixels on outer edge of image (those which cause the filter to be off the image) will be ignored
        # The outer two for loops move the filter central pixel over the image
        for row in range(floor_filter_height,height-floor_filter_height):
            for column in range(floor_filter_width,width-floor_filter_width):
                sum_positive = 0 # sum of positive coefficients
                sum_negative = 0 # sum of negative coefficients
                # These two for loops do the averaging within the filters bounds
                for j in range(-1*floor_filter_width, floor_filter_width+1): # moves across the columns of the filter
                    for i in range(-1*floor_filter_height, floor_filter_height+1): # moves down the rows of the filter
                        # difference filter defined as sum of positive filter coefficients minus sum of negative filter coefficients

                        # if filter weight is zero then no need to do anything
                        if weighted_filter[i+floor_filter_height][j+floor_filter_width] == 0:
                            continue
                        # weighted filter coefficient is positive
                        elif weighted_filter[i+floor_filter_height][j+floor_filter_width] > 0:
                            p = image[row+i][column+j] * weighted_filter[i+floor_filter_height][j+floor_filter_width]
                            sum_positive = sum_positive + p # sum all positive pixel coefficients
                        # weighted filter coefficient is negative
                        elif weighted_filter[i+floor_filter_height][j+floor_filter_width] < 0:
                            p = image[row+i][column+j] * abs(weighted_filter[i+floor_filter_height][j+floor_filter_width])
                            sum_negative = sum_negative + p # sum all negative pixel coefficients   

                # take differenece of positive and negative coefficients and store into image copy
                image_copy[row][column] = round(sum_positive-sum_negative)

        return image_copy # return altered image


    def medianOf2dImage(self,image,weighted_filter):
        """
        2D Image median with a user specified (via .env file) median filter
        
        Parameters:
        -----------
            image(numpy array): the image to be worked on

            weighted_filter : the weighted filter used to alter the image::

                dict:{
                        'filter': numpy matrix,
                        'column_vector': None(if not separable) or numpy array(if separable),
                        'row_vector': None(if not separable) or numpy array(if separable)
                    }

        Returns:
        --------
            image(numpy array): the altered image
        """

        # Getting the dimensions of the image
        height, width = image.shape
        image_copy= np.copy(image) # make copy of image

        filter_height, filter_width = weighted_filter.shape # get width and height of filter
        floor_filter_height = math.floor(filter_height/2)
        floor_filter_width = math.floor(filter_width/2)

        median_index = math.floor(weighted_filter.sum()/2) # median index value for pixel list

        # Pixels on outer edge of image (those which cause the filter to be off the image) will be ignored
        # The outer two for loops move the filter central pixel over the image
        for row in range(floor_filter_height,height-floor_filter_height):
            for column in range(floor_filter_width,width-floor_filter_width):
                pixel_list = []
                # These two for loops add pixels to list within the filters bounds
                for j in range(-1*floor_filter_width, floor_filter_width+1): # moves across the columns of the filter
                    for i in range(-1*floor_filter_height, floor_filter_height+1): # moves down the rows of the filter
                        # push pixel value into pixel_list number of times equal to the weight of that pixel
                        p = image[row+i][column+j] # the image pixel
                        num_p_elements = weighted_filter[i+floor_filter_height][j+floor_filter_width] # num times image pixel goes in list
                        for _ in range(num_p_elements):
                            pixel_list.append(p) # add weighted pixel to list

                pixel_list.sort() # sort list of pixels
                image_copy[row][column] = pixel_list[median_index] # store weighted median pixel back into image copy

        return image_copy # return averaged image
