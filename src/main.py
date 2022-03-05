from matplotlib import pyplot as plt
import numpy as np
import os
from dotenv import load_dotenv
import random
import glob
import math


def rgbToSingleChannels(image_path):
    """
    Loads an RGB image into a numpy array and divides the image into its red, green, blue channels.
    Red, green and blue channels are used to produce a greyscale channel as well using the formula
    grey_channel = .2989*red + .587*green + .114*blue

    Parameters:
        image_path: the filepath to the image to be opened and parsed into channels

    Returns:
        red_channel array, green_channel array, blue_channel array, grey_channel array
    """

    img = plt.imread(image_path) #load the image into a numpy array
    red_channel, green_channel, blue_channel = img[:,:,0], img[:,:,1], img[:,:,2] #separate three layers of the array into their RGB parts
    grey_channel = np.rint((0.2989 * red_channel) + (0.5870 * green_channel) + (0.1140 * blue_channel)) #use RGB to grayscale conversion formula
    return red_channel, green_channel, blue_channel, grey_channel


def createHistogram(image, bins=255, range=(0, 255)):
    """
    Performs histogram calculation (defaults to 255 bins) on a MxN numpy array of pixels with default range 0 to 255
    
    Parameters:
        image(numpy array): the image to build the histogram for
        bins(int): the number of bins for the histogram
        range(tuple): the range of the bins

    Returns:
        bin_values(numpy array): represents the number of pixels in each bin
        bins (numpy array): defines the bins which should be matched with histogram pixel values
    """

    # create the histogram
    bin_values, bins = np.histogram(image, bins, range)
    return bin_values, bins


def plotHistogram(bin_values, bins):
    """
    Plots histogram as a line graph and displays it

    Parameters:
        bin_values: the value of each bin
        bins: the definition of the bins

    Returns:
        None
    """

    plt.figure()
    plt.title("Image Histogram")
    plt.xlabel("pixel value")
    plt.ylabel("pixel count")
    plt.xlim([0.0, 255.0])
    plt.plot(bins[0:-1], bin_values)
    plt.show()


def addSaltAndPepperNoise(image, num_salt_pixels=100, num_pepper_pixels=100):
    """
    Randomly adds user specified number of salt and pepper pixels to an image

    Parameters:
        image(numpy array): the image to add noise to
        num_salt_pixels(int): number of 0 pixels to add
        num_pepper_pixels(int): number of 255 pixels to add

    Returns:
        image (numpy array): The altered image
    """

    # Getting the dimensions of the image
    row , col = image.shape

    if (num_salt_pixels < 0 or num_salt_pixels > image.size):
        print("Randomly setting number of salt pixels")
        num_salt_pixels = random.randint(0, image.size)
    if (num_pepper_pixels < 0 or num_pepper_pixels > image.size):
        print("Randomly setting number of pepper pixels")
        num_pepper_pixels = random.randint(0, image.size)
    
    # add user specified number of salt pixels at random positions
    for _ in range(num_salt_pixels):       
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)         
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)         
        # Color that pixel to white
        image[y_coord][x_coord] = 255
  
    # add user specified number of pepper pixels at random positions
    for _ in range(num_pepper_pixels):       
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)         
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)         
        # Color that pixel to black
        image[y_coord][x_coord] = 0
         
    return image # return the altered image


def addGaussianNoise(image,std_dev=1):
    """
    Adds gaussian noise to an image by randomly sampling values from a normal distribution
    centered around the specified mean and within the number of standard deviations. These
    random samples are then added to the original image to create a noisy image. 

    Parameters:
        mean(int): The value around which the gaussian distribution samples will be centered.
        std_dev(int): The number of standard deviation from the mean around which samples will be drawn.

    Returns:
        image_noisy: The corrupted image which is a sum of the original image and the random gaussian
        noisy distribution.
    """
    
    noise = np.random.normal(0, std_dev, size = image.shape)
    image_noisy = image + noise
    return image_noisy 


def getAllPathsInFolderByType():
    """
    Loads .env PATH_TO_IMAGES variable and IMAGE_EXTENSION variable and gets all filepaths contained within that folder and any folder
    below it by file extension

    Parameters:
        None

    Returns:
        imagepaths(list): the list of filepaths
    """

    file_extension = os.getenv('IMAGE_EXTENSION')
    path = os.path.join(os.getenv('PATH_TO_IMAGES'), f'**/*{file_extension}') # load .env path and concat with **/*{file_extension}
    imagepaths = glob.glob(path, recursive=True) # load all .BMP filepaths contained in images_path or any subfolder
    return imagepaths


def smoothGrayscaleImage(image, weighted_filter):
    """
    Image smoothing with a 3x3 averaging filter
    
    Parameters:
        image(numpy array): the image to be smoothed
        weighted_filter(numpy array): the weighted filter used to smooth the image

    Returns:
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
                    p = image_copy[row+i][column+j] * weighted_filter[i+floor_filter_height][j+floor_filter_width] 
                    sum = sum + p # sum all pixels within the neighborhood

            q = round(sum/weighted_filter.sum()) # divide by the number of pixels
            image[row][column] = q # store averaged pixel back into original image

    return image # return averaged image


def createFilterMatrix():
    """
    Creates filter matrix from that which is defined in the .env file. Turns matrix into a numpy array.

    Parameters:
        None

    Returns:
        array: filter as a 2D numpy array
    """

    filter = os.getenv('LINEAR_FILTER').splitlines() #split filter on newlines and place in list
    array = []
    # Each list item (other than the first) represents a row of the filter matrix    
    for row in range(1, len(filter)):
        filter_list = filter[row].split(',')
        array.append(list(map(int,filter_list))) #build 2D array

    # TODO: width and height of filter should both be odd values (probably should check for this)
    return np.asarray(array) # return as numpy array


if __name__ == "__main__":
    load_dotenv() # parse and load .env file
    filter = createFilterMatrix()

    imagepaths = getAllPathsInFolderByType() # get list of filepaths by file extension
    red_channel, green_channel, blue_channel, grey_channel = rgbToSingleChannels(imagepaths[0]) #load the image into a numpy array

    averaged_image = smoothGrayscaleImage(grey_channel, filter) # use averaging (smoothing) of a grayscale image

    #bin_values, bins = createHistogram(grey_channel)
    #plotHistogram(bin_values, bins)
    #noisy_image = addSaltAndPepperNoise(grey_channel, num_salt_pixels=1000, num_pepper_pixels=1000)
    #noisy_image = addGaussianNoise(grey_channel, std_dev=10)

    #plt.imsave('cell_images_original\cyl_cells\cyl01_modified.BMP', grey_channel, cmap='gray', vmin=0, vmax=255) #Save back grayscale image
    plt.subplot(1, 2, 1)
    plt.imshow(averaged_image, cmap='gray', vmin=0, vmax=255)
    plt.title('Averaged Image')
    plt.subplot(1, 2, 2)
    plt.imshow(grey_channel, cmap='gray', vmin=0, vmax=255)
    plt.title('Original')
    plt.show()
