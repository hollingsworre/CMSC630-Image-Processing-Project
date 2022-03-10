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
    compression_level (int) : quantization level defined in .env file
    save_images_path (str) : path to save images to defined in .env file
    color_spectrum (str) : spectrum of image to work on defined in .env file


    Methods
    -------

    getAllPathsInFolderByType(self)
        Loads .env PATH_TO_IMAGES variable and IMAGE_EXTENSION variable and gets all filepaths contained within that folder and any folder
        below it by file extension

    saveImage(self,image)
        Save image as grayscale

    rgbToSingleChannels(self,image_path)
        Loads an RGB image into a numpy array and divides the image into its red, green, blue channels.
        Red, green and blue channels are used to produce a greyscale channel as well using the formula
        grey_channel = .2989*red + .587*green + .114*blue

    showGrayscaleImages(self,images)
        Display up to four grayscale images at once.

    quantizeImage(self,image)
        Image compression into number of bins user specifies in the .env file

    quantizationError(self, original_image, uncompressed_image)
        Calculates and returns the MSQE for an uncompressed image in comparison to the original
    """

    def __init__(self):
        self.imagepaths = self.getAllPathsInFolderByType()
        self.compression_level = int(os.getenv('COMPRESSION_LEVEL'))
        self.save_images_path = os.getenv('SAVE_IMAGES_PATH')
        self.color_spectrum=os.getenv('COLOR_SPECTRUM').lower()


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


    def saveImage(self,image,image_pathname):
        """
        Save image as grayscale

        Parameters:
        -----------
            image (numpy array) : image to be saved
            image_pathname (str) : pathname of image

        Returns:
        --------
            None
        """
        filename = 'modified_' + os.path.basename(image_pathname)   # extract filename of image from its pathname and add modified_
        path = os.path.join(self.save_images_path, filename)        # join save path and filename

        plt.imsave(path, image, cmap='gray', vmin=0, vmax=255) #Save back grayscale image
    


    def rgbToSingleChannels(self,image_path):
        """
        Loads an RGB image into a numpy array and divides the image into its red, green, blue channels.
        Red, green and blue channels are used to produce a greyscale channel as well using the formula
        grey_channel = .2989*red + .587*green + .114*blue

        Parameters:
        -----------
            image_path : the filepath to the image to be opened and parsed into channels

        Returns:
        --------
            channel (numpy array) : red, green, blue, or grey depending on the color spectrum requested in the .env

        """

        channels = {'red':None,'green':None,'blue':None,'grey':None}

        img = plt.imread(image_path) #load the image into a numpy array
        channels['red'], channels['green'], channels['blue'] = img[:,:,0], img[:,:,1], img[:,:,2] #separate three layers of the array into their RGB parts
        channels['grey'] = np.rint((0.2989 * channels['red']) + (0.5870 * channels['green']) + (0.1140 * channels['blue'])) #use RGB to grayscale conversion formula
        
        # return requested channel
        if self.color_spectrum in channels:
            return channels[self.color_spectrum]
        # requested channel does not exist so return grey channel
        else:
            return channels['grey']


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
        Image compression into number of intensity levels the user specifies in the .env file.

        Parameters:
        -----------
            image(numpy array) : the image to compress

        Returns:
            compressed_image(numpy array) : the image after compression
        """
        # Flatten the image
        image_list = list(image.flatten())
        # Get range of pixel intensities within the image
        #range = max(image_list)-min(image_list)
        intensity_range = 255 - 0 # lmax - lmin
        compression_factor = intensity_range/self.compression_level
        compressed_image_list = []

        # compress pixel into it's appropriate new intensity value
        for pixel in image_list:            
            compressed_image_list.append(math.floor(pixel/compression_factor))

        compressed_image = np.reshape(np.asarray(compressed_image_list), image.shape)

        return compressed_image


    def decompressImage(self, compressed_image):
        """
        Decompress an image that was compressed via the quantizeImage function in this class.
        In order to decompress, you must know the max - min pixel range (255 here) from the original
        as well as the number of bins the original image was compressed down into.
        
        uncompressed_pixel == (compressed_pixel * intensity_range_original_image)/(compressed_bin_range)

        Parameters:
        -----------
            compressed_image(numpy array) : the image to decompress

        Returns:
        --------
            decompressed_image(numpy array) : the image decompressed
        """
        # Flatten the image
        compressed_image_list = list(compressed_image.flatten())
        intensity_range = 255 - 0 # lmax - lmin
        decompressed_image_list = []

        # compress pixel into it's appropriate new intensity value
        for pixel in compressed_image_list:            
            decompressed_image_list.append(math.floor((pixel*intensity_range)/self.compression_level))

        decompressed_image = np.reshape(np.asarray(decompressed_image_list), compressed_image.shape)

        return decompressed_image


    def quantizationError(self, original_image, uncompressed_image):
        """
        Calculates and returns the MSQE for an uncompressed image in comparison to the original

        Parameters:
        -----------
            original_image(numpy_array) : the original uncompressed image
            uncompressed_image(numpy_array) : the image after compression (quantizeImage() function)
            and decompression (decompressImage() function)

        Returns:
        --------
            msqe(float) : the mean square error of the two images
        """

        # Sum of errors squared divided by image size
        msqe = (np.square(np.subtract(original_image,uncompressed_image)).sum())/(original_image.size)
        return msqe
