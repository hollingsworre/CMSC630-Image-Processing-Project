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
        print(bins)
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
        