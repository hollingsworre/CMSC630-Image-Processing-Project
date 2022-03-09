from matplotlib import pyplot as plt
import numpy as np

class Histogram:
    """
    Component class for all histogram operations.

    Attributes
    ----------

    histograms_sums : Dictionary used for calculating the sum of histogram pixel values for each image class::

        dict : { 'cyl' : {'HistogramSum'(numpy array):None, 'TotalHistograms'(int):0, 'bins':None},
                 'inter' : {'HistogramSum'(numpy array):None, 'TotalHistograms'(int):0, 'bins':None},
                 'let' : {'HistogramSum'(numpy array):None, 'TotalHistograms'(int):0, 'bins':None},
                 'mod' : {'HistogramSum'(numpy array):None, 'TotalHistograms'(int):0, 'bins':None},
                 'para' : {'HistogramSum'(numpy array):None, 'TotalHistograms'(int):0, 'bins':None},
                 'super' : {'HistogramSum'(numpy array):None, 'TotalHistograms'(int):0, 'bins':None},
                 'svar' : {'HistogramSum'(numpy array):None, 'TotalHistograms'(int):0, 'bins':None}
                }

    averaged_histograms : averaged histograms after averageHistogramsByType() is called::

        dict : { 'cyl' : (numpy array),
                 'inter' : (numpy array),
                 'let' : (numpy array),
                 'mod' : (numpy array),
                 'para' : (numpy array),
                 'super' : (numpy array),
                 'svar' : (numpy array)
                }

    Methods
    -------

    createHistogram(self,image,bins=255,range=(0, 255))
        Performs histogram calculation (defaults to 255 bins) on a MxN numpy array of pixels with default range 0 to 255

    createAndPlotHistograms(self,images,num_rows=1,num_cols=1)
        Create and plot histograms of multiple images as line graphs and display them.

    histogramEqualization(self,image,display_histograms=False)
        Performs histogram equalization on an individual image by using the pixel's
        probability distribution within the image for automatic stretching or compression.

    sumHistogramsByType(self,image_path,bin_values,bins)
        Function called from createHistogram function and used for the summing
        of histograms by type.

    averageHistogramsByType(self)
        Averages all histograms in self.histogram_sums dictionary

    plotAveragedHistogramsByType(self,num_rows=4,num_cols=4)
        Plots self.averaged_histograms
    """

    def __init__(self):
        # Dictionary used for calculating the sum of histogram pixel values for each image class
        self.histogram_sums = { 'cyl' : {'HistogramSum':None, 'TotalHistograms':0, 'bins':None},
                                'inter' : {'HistogramSum':None, 'TotalHistograms':0, 'bins':None},
                                'let' : {'HistogramSum':None, 'TotalHistograms':0, 'bins':None},
                                'mod' : {'HistogramSum':None, 'TotalHistograms':0, 'bins':None},
                                'para' : {'HistogramSum':None, 'TotalHistograms':0, 'bins':None},
                                'super' : {'HistogramSum':None, 'TotalHistograms':0, 'bins':None},
                                'svar' : {'HistogramSum':None, 'TotalHistograms':0, 'bins':None}
                                }

        self.averaged_histograms = None


    def createHistogram(self,image,bins=256,range=[0, 256],image_path=None):
        """
        Performs histogram calculation (defaults to 255 bins) on a MxN numpy array of pixels with default range 0 to 255.
        If image path is supplied then that image's histogram will be sent to the averageHistogramsByType function.
        
        Parameters:
        -----------
            image(numpy array): the image to build the histogram for
            bins(int): the number of bins for the histogram
            range(tuple): the range of the bins
            image_path(str) : the path to the image to be processed

        Returns:
        --------
            bin_values(numpy array): represents the number of pixels in each bin
            bins (numpy array): defines the bins which should be matched with histogram pixel values
        """

        # create the histogram
        bin_values, bins = np.histogram(image, bins, range)

        # if path is present, then that means histograms are to be averaged by type as well
        if image_path is not None:
            self.sumHistogramsByType(image_path,bin_values,bins)

        return bin_values, bins


    def createAndPlotHistograms(self,images,num_rows=1,num_cols=1,num_bins=256,bin_range=[0,256],x_axis_limit=255.0):
        """
        Create and plot histograms of multiple images as line graphs and display them.

        Parameters:
        -----------
            images(list): list of images to display
            num_rows(int): number of rows to be displayed in plot
            num_cols(int): number of columns to be displayed in plot
            num_bins(int): the number of bins for the histogram
            bin_range(tuple): the range of the bins
            x_axis_limit(float): limit of the x axis on the plots

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
            plt.xlim([0.0, x_axis_limit])
            bin_values, bins = self.createHistogram(images[i],bins=num_bins,range=bin_range)
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


    def sumHistogramsByType(self,image_path,bin_values,bins):
        """
        Function called from createHistogram function and used for the summing
        of histograms by type.

        Parameters:
        -----------
            image_path(str) : path where image is located
            bin_values (array) : The bin values (num pixels per intensity level defined by bins) for the image
            bins (array) : The bins of the image

        Returns:
            None
        """

        # figure out which key image path belongs to
        for i in self.histogram_sums:
            if i in image_path:
                # increment number of that type of histogram
                self.histogram_sums[i]['TotalHistograms'] = self.histogram_sums[i]['TotalHistograms'] + 1
                if self.histogram_sums[i]['HistogramSum'] is None:
                    self.histogram_sums[i]['HistogramSum'] = bin_values
                    self.histogram_sums[i]['bins'] = bins
                else:
                    self.histogram_sums[i]['HistogramSum'] = np.add(self.histogram_sums[i]['HistogramSum'], bin_values)


    def averageHistogramsByType(self):
        """
        Averages all histograms in self.histogram_sums dictionary

        Parameters:
            None

        Returns:
            averaged_histogram(numpy array) : all averaged histograms for each image type found
        """
        averaged_histograms = {}
        for i in self.histogram_sums:
            if self.histogram_sums[i]['HistogramSum'] is not None:
                averaged_histograms[i] = np.rint(self.histogram_sums[i]['HistogramSum']/self.histogram_sums[i]['TotalHistograms'])

        # Save averaged histograms into class variable
        self.averaged_histograms = averaged_histograms

        return averaged_histograms


    def plotAveragedHistogramsByType(self,num_rows=2,num_cols=4):
        """
        Plots self.averaged_histograms

        Parameters:
        -----------
            num_rows(int) : number of rows in the plot
            num_cols(int) : number of columns in the plot

        Returns:
        --------
            None
        """

        if self.averaged_histograms is not None:
            j = 1
            for i in self.averaged_histograms:
                plt.subplot(num_rows, num_cols, j)
                plt.title(f'{i}')
                plt.xlim([0.0, 255.0])
                bin_values = self.averaged_histograms[i]
                bins = self.histogram_sums[i]['bins']
                plt.plot(bins[0:-1], bin_values)
                j = j+1

            plt.show()
