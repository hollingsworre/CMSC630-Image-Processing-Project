import os
import glob
from matplotlib import pyplot as plt
import numpy as np
import math


class Images:
    """
    Component class for all images to be worked on.

    Attributes
    ----------

    imagepaths (list) : list of all image paths found which should be loaded and worked on
    

    Methods
    -------

    rgbToSingleChannels(self,image_path)
        Loads an RGB image into a numpy array and divides the image into its red, green, blue channels.
        Red, green and blue channels are used to produce a greyscale channel as well using the formula
        grey_channel = .2989*red + .587*green + .114*blue

    showGrayscaleImages(self,images)
        Display up to four grayscale images at once.

    quantizeImage(self,image)
        Image compression into number of bins user specifies in the .env file
    """

    def __init__(self):
        self.imagepaths = self.getAllPathsInFolderByType()
        self.compression_level = int(os.getenv('COMPRESSION_LEVEL'))


    def getAllPathsInFolderByType(self):
        """
        Loads .env PATH_TO_IMAGES variable and IMAGE_EXTENSION variable and gets all filepaths contained within that folder and any folder
        below it by file extension

        Parameters:
        -----------
            None

        Returns:
        --------
            imagepaths(list): the list of filepaths
        """

        file_extension = os.getenv('IMAGE_EXTENSION')
        path = os.path.join(os.getenv('PATH_TO_IMAGES'), f'**/*{file_extension}') # load .env path and concat with **/*{file_extension}
        imagepaths = glob.glob(path, recursive=True) # load all .BMP filepaths contained in images_path or any subfolder
        return imagepaths


    def rgbToSingleChannels(self,image_path):
        """
        Loads an RGB image into a numpy array and divides the image into its red, green, blue channels.
        Red, green and blue channels are used to produce a greyscale channel as well using the formula
        grey_channel = .2989*red + .587*green + .114*blue

        Parameters:
        -----------
            image_path: the filepath to the image to be opened and parsed into channels

        Returns:
        --------
            red_channel array, green_channel array, blue_channel array, grey_channel array
        """

        img = plt.imread(image_path) #load the image into a numpy array
        red_channel, green_channel, blue_channel = img[:,:,0], img[:,:,1], img[:,:,2] #separate three layers of the array into their RGB parts
        grey_channel = np.rint((0.2989 * red_channel) + (0.5870 * green_channel) + (0.1140 * blue_channel)) #use RGB to grayscale conversion formula
        return red_channel, green_channel, blue_channel, grey_channel


    def showGrayscaleImages(self,images,num_rows=1,num_cols=1):
        """
        Display specified number of grayscale images at once.

        Parameters:
        -----------
            images(list): list of images to display
            num_rows(int): number of rows to be displayed in plot
            num_cols(int): number of columns to be displayed in plot

        Returns:
        -------
            None
        """

        for i in range(len(images)):
            if i == 4: # only allowed to show four images at once
                break
            plt.subplot(num_rows, num_cols, i+1)
            plt.imshow(images[i], cmap='gray', vmin=0, vmax=255)
            plt.title(f'Image {i}')

        plt.show()


    def quantizeImage(self,image):
        """
        Image compression into number of intensity levels the user specifies in the .env file

        Parameters:
        -----------
            image(numpy array) : the image to compress

        Returns:
            compressed_image(numpy array) : the image after compression
        """
        # Flatten the image
        image_list = list(image.flatten())
        # Get range of pixel intensities within the image
        range = max(image_list)-min(image_list)
        compression_factor = range/self.compression_level
        compressed_image_list = []

        # compress pixel into it's appropriate new intensity value
        for pixel in image_list:            
            compressed_image_list.append(math.floor(pixel/compression_factor))

        compressed_image = np.reshape(np.asarray(compressed_image_list), image.shape)

        return compressed_image


    def decompressImage(self, image):
        """
        Decompress an image that was compressed via the quantizeImage function in this class.
        In order to decompress, you must know the max - min pixel range from the original.
        
        uncompressed_pixel == (compressed_pixel * compressed_bin_range)/(intensity_range_original_image)
        """

