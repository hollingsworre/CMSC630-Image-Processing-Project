from matplotlib import pyplot as plt
import numpy as np

class Histogram:
    """
    Component class for all histogram operations.

    Methods
    -------

    createHistogram(self,image,bins=255,range=(0, 255))
        Performs histogram calculation (defaults to 255 bins) on a MxN numpy array of pixels with default range 0 to 255

    createAndPlotHistograms(self,images,num_rows=1,num_cols=1)
        Create and plot histograms of multiple images as line graphs and display them.

    histogramEqualization(self,image,display_histograms=False)
        Performs histogram equalization on an individual image by using the pixel's
        probability distribution within the image for automatic stretching or compression.
    """

    def __init__(self):
        pass


    def createHistogram(self,image,bins=256,range=[0, 256]):
        """
        Performs histogram calculation (defaults to 255 bins) on a MxN numpy array of pixels with default range 0 to 255
        
        Parameters:
        -----------
            image(numpy array): the image to build the histogram for
            bins(int): the number of bins for the histogram
            range(tuple): the range of the bins

        Returns:
        --------
            bin_values(numpy array): represents the number of pixels in each bin
            bins (numpy array): defines the bins which should be matched with histogram pixel values
        """

        # create the histogram
        bin_values, bins = np.histogram(image, bins, range)
        return bin_values, bins


    def createAndPlotHistograms(self,images,num_rows=1,num_cols=1):
        """
        Create and plot histograms of multiple images as line graphs and display them.

        Parameters:
        -----------
            images(list): list of images to display
            num_rows(int): number of rows to be displayed in plot
            num_cols(int): number of columns to be displayed in plot

        Returns:
        --------
            None
        """

        for i in range(len(images)):
            if i == 4: # only allowed to show four images at once
                break
            plt.subplot(num_rows, num_cols, i+1)
            plt.title(f'Image {i}')
            #plt.xlabel("pixel value")
            plt.ylabel("pixel count")
            plt.xlim([0.0, 255.0])
            bin_values, bins = self.createHistogram(images[i])
            plt.plot(bins[0:-1], bin_values)

        plt.show()

    
    def histogramEqualization(self,image):
        """
        Performs histogram equalization on an individual image by using the pixel's
        probability distribution within the image for automatic stretching or compression.
        
        Parameters:
        ----------
            image(numpy array) : the image to be equalized

        Returns:
        --------
            equalized_image(numpy array) : the equalized image
        """

        num_pixels = image.size # get total number of pixels within the image       
        bin_values, bins = self.createHistogram(image) # get image histogram
        num_bins = len(bins)-1 # get total number of bins  
        equalized_bin_values = [] # bin values after histogram is equalized
        current_sum = 0
        
        # Create pixel mapping lookup table
        for i in range(num_bins):
            current_sum = current_sum + bin_values[i]
            equalized_bin_values.append(round((current_sum*255)/num_pixels))

        # Create equalized image
        image_list = list(image.astype(int).flatten())
        equalized_image_list = []
        for i in image_list:
            equalized_image_list.append(equalized_bin_values[i])

        equalized_image = np.reshape(np.asarray(equalized_image_list), image.shape)
        
        return equalized_image
