from matplotlib import pyplot as plt
import numpy as np

class Histogram:
    """
    Component class for all histogram operations.

    Methods
    -------

    createHistogram(self,image,bins=255,range=(0, 255))
        Performs histogram calculation (defaults to 255 bins) on a MxN numpy array of pixels with default range 0 to 255

    plotHistogram(bin_values,bins)
        Plots histogram as a line graph and displays it
    """

    def __init__(self):
        pass


    def createHistogram(self,image,bins=255,range=(0, 255)):
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


    def plotHistogram(bin_values,bins):
        """
        Plots histogram as a line graph and displays it

        Parameters:
        -----------
            bin_values: the value of each bin
            bins: the definition of the bins

        Returns:
        --------
            None
        """

        plt.figure()
        plt.title("Image Histogram")
        plt.xlabel("pixel value")
        plt.ylabel("pixel count")
        plt.xlim([0.0, 255.0])
        plt.plot(bins[0:-1], bin_values)
        plt.show()
        