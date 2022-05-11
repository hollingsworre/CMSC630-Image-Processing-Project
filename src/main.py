from dotenv import load_dotenv
import os
import time
import multiprocessing
import csv
import numpy as np
import math
from components.filters import Filters
from components.point_operations import ImagePointOperations
from components.images import Images
from components.noise import Noise
from components.histogram import Histogram
from components.segmentation import Segmentation
from components.edges import Edges
from components.knn_classifier import KNN_Classifer


class Main:
    """
    Composite class made up of all components.

    Attributes
    ----------

        filters(obj) : filters component plugin instance
        point_operations(obj) : point_operations component plugin instance
        images(obj) : images component plugin instance
        noise_functions(obj) : noise_functions component plugin instance
        histogram_functions(obj) : histogram component plugin instance
        segmentation(obj) : segmentation component plugin instance
        edges(obj) : edge detection component plugin instance
        timing_results(list) : list for storing timing results of each image operation
        msqe_results(list) : list for storing msqe numbers per image
        equalization_msqe(float) : msqe calculation for histogram equalization
        function_dictionary : Dictionary of function pointers used for batch processing of user specified
                             functions from the .env::

                                    dict:{
                                            '1':(function) add_salt_pepper_noise,
                                            '2':(function) add_gaussian_noise,
                                            '3':(function) histogram_equalization,
                                            '4':(function) histogram_quantization,
                                            '5':(function) box_smoothing,
                                            '6':(function) gaussian_smoothing,
                                            '7':(function) lapacian_difference,
                                            '8':(function) median_smoothing,
                                            '9':(function) create_average_histograms,
                                            '10':(function) k_means_segmentation,
                                            '11':(function) histogram_thresholding_segmentation,
                                            '12':(function) edge_erosion,
                                            '13':(function) edge_dilation,
                                            '14':(function) edge_detection,
                                            '17':(function) feature_extraction
                                            '18':(function) train_classifier
                                            }

    Methods
    -------

        save_data(self, result)
            Callback method for the process pool started up in self.run_batch_mode().
            Appends the processing time for the individual process into self.timing_results.

        parallelModel(self, path, function_list)
            Method for performing image operations which can be parallelized.

        run_batch_mode(self)
            Batch mode function. If user is requesting an averaged histogram operation then
            this will be performed synchronously by a single process. Otherwise a process pool equal to the 
            number of cpus on the machine will be spun up and the filepaths will be distributed to the processes
            for parallelization. Otherwise it takes a LONG time to process any significant number of images.
    """

    def __init__(self):        
        # component classes that can be added
        self.filters = None
        self.point_operations = None
        self.images = None
        self.noise_functions = None
        self.histogram_functions = None
        self.segmentation = None
        self.edges = None
        self.classifier = None
        self.timing_results = []
        self.msqe_results = []
        self.feature_fields = ['Nucleus Area','Nucleus Perimeter','Nucleus Roundness','Cell Area','Label']
        self.feature_rows = []
        self.function_dictionary = {'1':self.add_salt_pepper_noise,
                                    '2':self.add_gaussian_noise,
                                    '3':self.histogram_equalization,
                                    '4':self.histogram_quantization,
                                    '5':self.box_smoothing,
                                    '6':self.gaussian_smoothing,
                                    '7':self.lapacian_difference,
                                    '8':self.median_smoothing,
                                    '9':self.create_average_histograms,
                                    '10':self.k_means_segmentation,
                                    '11':self.histogram_thresholding_segmentation,
                                    '12':self.edge_erosion,
                                    '13':self.edge_dilation,
                                    '14':self.edge_detection,
                                    '17':self.feature_extraction,
                                    '18':self.train_classifier}


    def save_data(self, result):
        """
        Callback method for the process pool started up in self.run_batch_mode().
        Appends the processing time for the individual process into self.timing_results.

        Parameters:
        -----------
            result(array) : [processing time for the process, msqe for the quantization, feature_list]

        Returns:
        --------
            None
        """
        self.timing_results.append(result[0])
        if result[1] != 0:
            self.msqe_results.append(result[1])
        if result[2]:            
            self.feature_rows.append(result[2]) # push features and label into csv list


    def add_salt_pepper_noise(self,image):
        """
        Calls component function for adding of salt and pepper noise.
        
        Parameters:
        -----------
            image(numpy 2D array) : grayscale image to be worked on

        Returns:
        --------
            image(numpy 2D array) : the altered grayscale image
        """
        return self.noise_functions.addSaltAndPepperNoise(image)

    def add_gaussian_noise(self,image):
        """
        Calls componenet function for adding of gaussian noise.
        
        Parameters:
        -----------
            image(numpy 2D array) : grayscale image to be worked on

        Returns:
        --------
            image(numpy 2D array) : the altered grayscale image
        """
        return self.noise_functions.addGaussianNoise(image)

    def histogram_equalization(self,image):
        """
        Calls component function for histogram equalization.
        
        Parameters:
        -----------
            image(numpy 2D array) : grayscale image to be worked on

        Returns:
        --------
            image(numpy 2D array) : the equalized grayscale image
        """
        return self.histogram_functions.histogramEqualization(image)

    def histogram_quantization(self,image):
        """
        Calls component functions for quantization, decompression and quantization error calculations.
        
        Parameters:
        -----------
            image(numpy 2D array) : grayscale image to be worked on

        Returns:
        --------
            equalization_msqe(float) : msqe after compression and decompression of the grayscale image
        """
        altered_image = self.images.quantizeImage(image)
        decompressed_image = self.images.decompressImage(altered_image)
        equalization_msqe = self.images.quantizationError(image, decompressed_image)
        return equalization_msqe

    def box_smoothing(self,image):
        """
        Calls component function smoothing of an image with a box filter.
        
        Parameters:
        -----------
            image(numpy 2D array) : grayscale image to be worked on

        Returns:
        --------
            image(numpy 2D array) : the altered grayscale image
        """
        return self.point_operations.smooth2dImage(image, self.filters.box_filter['filter'])

    def gaussian_smoothing(self,image):
        """
        Calls component function for smoothing an image with a gaussian filter.
        
        Parameters:
        -----------
            image(numpy 2D array) : grayscale image to be worked on

        Returns:
        --------
            image(numpy 2D array) : the altered grayscale image
        """
        return self.point_operations.smooth2dImage(image, self.filters.gaussian_filter['filter'])

    def lapacian_difference(self,image):
        """
        Calls component function for laplacian difference on an image.
        
        Parameters:
        -----------
            image(numpy 2D array) : grayscale image to be worked on

        Returns:
        --------
            image(numpy 2D array) : the altered grayscale image
        """
        return self.point_operations.difference2dImage(image, self.filters.laplacian_filter['filter'])

    def median_smoothing(self,image):
        """
        Calls component function for median filtering of an image.
        
        Parameters:
        -----------
            image(numpy 2D array) : grayscale image to be worked on

        Returns:
        --------
            image(numpy 2D array) : the altered grayscale image
        """
        return self.point_operations.medianOf2dImage(image, self.filters.median_filter['filter'])

    def create_average_histograms(self):
        """
        Averages all histograms by filetype and then plots them. Runs synchronously on a single thread.
        
        Parameters:
        -----------
            None

        Returns:
        --------
            None
        """
        for path in self.images.imagepaths:
            start_time = time.time()
            image = self.images.getImage(path,color_spectrum=self.images.color_spectrum)
            self.histogram_functions.createHistogram(image,image_path=path)
            self.timing_results.append(time.time() - start_time)
        self.histogram_functions.averageHistogramsByType()
        self.histogram_functions.plotAveragedHistogramsByType()

    def k_means_segmentation(self,image):
        """
        Calls component function for k-means segmentation of an rgb image.
        
        Parameters:
        -----------
            image(numpy 3D array) : rgb image to be segmented

        Returns:
        --------
            image(numpy 3D array) : the segmented rgb image
        """
        return self.segmentation.k_means_segmentation(image)

    def histogram_thresholding_segmentation(self,image):
        """
        Calls component functions for histogram creation and then segmentation by histogram thresholding.
        
        Parameters:
        -----------
            image(numpy 2D array) : grayscale image to be worked on

        Returns:
        --------
            image(numpy 2D array) : the segmented grayscale image
        """
        bin_values, bins = self.histogram_functions.createHistogram(image)
        return self.segmentation.histogram_thresholding_segmentation(image,bin_values,bins)

    def edge_detection(self,image,detection_type):
        """
        Calls component function for edge detection of a grayscale image.
        
        Parameters:
        -----------
            image(numpy 2D array) : grayscale image to be worked on
            detection_type (str) : type of edge dection to perform. Should be one of 'improved_sobel', 'sobel', 'prewitt'

        Returns:
        --------
            image(numpy 2D array) : the image with only edges marked
        """
        return self.edges.edge_detection(image,detection_type=detection_type,threshold=self.edges.edge_detection_threshold)

    def edge_erosion(self,image):
        """
        Calls component function for erosion of an edge image.
        
        Parameters:
        -----------
            image(numpy 2D array) : grayscale image to be worked on

        Returns:
        --------
            image(numpy 2D array) : the edge image with requested layers removed
        """
        return self.edges.edge_erosion(image,num_layers=self.edges.num_erosion_layers,structuring_element=self.filters.edge_erosion_element)

    def edge_dilation(self,image):
        """
        Calls component function for dilation of an edge image.
        
        Parameters:
        -----------
            image(numpy 2D array) : grayscale image to be worked on

        Returns:
        --------
            image(numpy 2D array) : the edge image with layers added
        """
        return self.edges.edge_dilation(image,num_layers=self.edges.num_dilation_layers,structuring_element=self.filters.edge_dilation_element)


    def feature_extraction(self,image,path):
        """Extracts features from a segmented greyscale image"""

        features = []   # ['Nucleus Area','Nucleus Perimeter','Nucleus Roundness','Cell Area']

        label_list = ['cyl','inter','let','mod','para','super','svar']
        image_label = ''
        # get label
        for label in label_list:
            if label in path:
                image_label = label
                break

        image_copy = np.copy(image)

        # get image histogram
        bin_values, _ = self.histogram_functions.createHistogram(image)

        # Ensure image is segmented properly
        if np.count_nonzero(bin_values) == self.segmentation.num_cluster_centers:
            pixel_index = np.nonzero(bin_values)[0] # [darkest -> lightest], e.g. nucleus -> cell -> background
            pixel_count = bin_values[np.nonzero(bin_values)] # get exact pixel counts, e.g. nucleus -> cell -> background
            nucleus_area = pixel_count[0] # calculate nucleus area
            cell_area = pixel_count[1] # calculate cell area
            image_copy[image != pixel_index[0]] = 255 # set everything but the nucleii to white in the image_copy
            # get the nucleii edges for perimeter calcs
            nucleus_edges = self.edges.edge_detection(image_copy,detection_type='improved_sobel',threshold=self.edges.edge_detection_threshold) 
            bin_values, _ = self.histogram_functions.createHistogram(nucleus_edges)  # get nucleus edges histogram
            edge_pixel_count = bin_values[np.nonzero(bin_values)] # get exact pixel counts, e.g. nucleus_edge -> background
            nucleus_perimeter = edge_pixel_count[0] # calculate nucleus perimeter
            nucleus_roundness = (nucleus_perimeter**2)/(4*math.pi*nucleus_area) # calculate nucleus roundness
            features = [nucleus_area, nucleus_perimeter, nucleus_roundness, cell_area, image_label]

        return features # return original segmented image


    def train_classifier(self):
        """Run knn classification algorithm"""
        start_time = time.time()
        self.classifier.run_k_fold_knn()
        self.timing_results.append(time.time() - start_time)


    def parallelModel(self, path, function_list):
        """
        Method for performing image operations which can be parallelized.

        Parameters:
        -----------
            path(str) : The path of the image
            function_list(list) : List of functions to be run from the .env file. All functions mapped in self.function_dictionary.

        Returns:
        --------
            processing_time(float) : The processing time for the operation
            equalization_msqe(float) : The msqe for the quantization
        """        
        
        equalization_msqe = 0
        features = []
        current_process = multiprocessing.Process().name
        print(f"{current_process} : processing image {path}")
        start_time = time.time()
        # get channel defined in .env file
        image = self.images.getImage(path,color_spectrum=self.images.color_spectrum)

        # apply all requested functions
        for i in function_list:            
            if i not in ['10']:
                # if k-means was the previous function then image will be 3d and
                # it needs to be converted to greyscale
                if len(image.shape) == 3:
                    image = self.images.rgbToGrayscale(image)

            # This is histogram quantization and msqe will be returned (not the image)
            if i in ['4']:              
                equalization_msqe = self.function_dictionary[i](image)
            # If k-means then it needs to be color image
            elif i in ['10']:
                # get image again if it is not rgb (K-means must be done in color)
                if len(image.shape) != 3:
                    image = self.images.getImage(path,color_spectrum='rgb')
                image = self.function_dictionary[i](image)
            # Could be one of three edge detection functions
            elif i in ['14','15','16']:
                mapping = {'14':'sobel','15':'improved_sobel','16':'prewitt'}
                image = self.function_dictionary['14'](image,mapping[i])
            # Feature extraction (need to pass in path so label can be created)
            elif i in ['17']:
                features = self.function_dictionary[i](image,path)
            else:
                image = self.function_dictionary[i](image)

        if 'image' in locals():
            self.images.saveImage(image,path) # save image as grayscale

        print(f"{current_process} : done with image {path}")

        # return the image processing time and equalization_msqe
        return [time.time() - start_time, equalization_msqe, features]


    def run_batch_mode(self):
        """
        Batch mode function. If user is requesting an averaged histogram operation then
        this will be performed synchronously by a single process. Otherwise a process pool equal to the 
        number of cpus on the machine will be spun up and the filepaths will be distributed to the processes
        for parallelization. Otherwise it takes a LONG time to process any significant number of images.

        Parameters:
        -----------
            None

        Returns:
        --------
            None
        """

        # retrieve list of functions to run on batch from .env file
        function_list = os.getenv('FUNCTION_LIST').split(",")
        function_list = [item.strip() for item in function_list] # strip out any whitespace

        if '9' in function_list:
            print('Averaging histograms and exiting. Do not include function 9 if you wish to run any others!')
            self.function_dictionary['9']()
        elif '18' in function_list:
            print('Training KNN classifier')
            self.function_dictionary['18']()
        else:
            # Create your process pool equal to the number of cpus detected on your machine
            pool = multiprocessing.Pool(os.cpu_count())
            # Use imagepaths iterable to dispatch paths to the process pool
            # Callback is used for storing the time of each image operation for final results at the end
            _ = [pool.apply_async(self.parallelModel, callback=self.save_data, args=(path,function_list)) for path in self.images.imagepaths]
            pool.close()
            pool.join()


if __name__ == "__main__":
    # parse and load .env file into the environment
    load_dotenv()
    
    # Instantiate composite class
    composite = Main()    

    # Add component classes to composite class
    composite.filters = Filters()
    composite.point_operations = ImagePointOperations()
    composite.images = Images()
    composite.noise_functions = Noise()
    composite.histogram_functions = Histogram()
    composite.segmentation = Segmentation()
    composite.edges = Edges()
    composite.classifier = KNN_Classifer()

    start_time = time.time()

    composite.run_batch_mode()

    print("\n--- Batch Processing Time: %s seconds ---" % (time.time() - start_time))
    
    average_processing_time = sum(composite.timing_results)/len(composite.timing_results)
    print("--- Processing Time Per Image: %s seconds ---\n" % (average_processing_time))
    
    if composite.msqe_results:
        average_msqe = sum(composite.msqe_results)/len(composite.msqe_results)
        print("--- Average MSQE: %s ---\n" % (average_msqe))

    # write extracted features to file
    if composite.feature_rows:
        with open('datasets\extracted_features.csv', 'w+', newline="") as f:
            write = csv.writer(f)            
            write.writerow(composite.feature_fields)
            write.writerows(composite.feature_rows) 

    # display up to four images at once
    #composite.images.showGrayscaleImages([salt_pepper_noise_image, filtered_sp_image], num_rows=1, num_cols=2)

    # Create and plot up to four histograms from a list of images
    #composite.histogram_functions.createAndPlotHistograms([grey_channel, salt_pepper_noise_image, filtered_sp_image], num_rows=2, num_cols=2)
