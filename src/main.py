from dotenv import load_dotenv
import os
import time
import multiprocessing
from components.filters import Filters
from components.point_operations import ImagePointOperations
from components.images import Images
from components.noise import Noise
from components.histogram import Histogram
from components.segmentation import Segmentation
from components.edges import Edges


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
        self.timing_results = []
        self.msqe_results = []
        self.equalization_msqe = 0
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
                                    '14':self.edge_detection}


    def save_data(self, result):
        """
        Callback method for the process pool started up in self.run_batch_mode().
        Appends the processing time for the individual process into self.timing_results.

        Parameters:
        -----------
            result(array) : [processing time for the process, msqe for the quantization]

        Returns:
        --------
            None
        """
        self.timing_results.append(result[0])
        if result[1] != 0:
            self.msqe_results.append(result[1])


    def add_salt_pepper_noise(self,image):
        return self.noise_functions.addSaltAndPepperNoise(image)

    def add_gaussian_noise(self,image):
        return self.noise_functions.addGaussianNoise(image)

    def histogram_equalization(self,image):
        return self.histogram_functions.histogramEqualization(image)

    def histogram_quantization(self,image):
        altered_image = self.images.quantizeImage(image)
        decompressed_image = self.images.decompressImage(altered_image)
        self.equalization_msqe = self.images.quantizationError(image, decompressed_image)

    def box_smoothing(self,image):
        return self.point_operations.smooth2dImage(image, self.filters.box_filter['filter'])

    def gaussian_smoothing(self,image):
        return self.point_operations.smooth2dImage(image, self.filters.gaussian_filter['filter'])

    def lapacian_difference(self,image):
        return self.point_operations.difference2dImage(image, self.filters.laplacian_filter['filter'])

    def median_smoothing(self,image):
        return self.point_operations.medianOf2dImage(image, self.filters.median_filter['filter'])

    def create_average_histograms(self):
        for path in self.images.imagepaths:
            start_time = time.time()
            image = self.images.getImage(path,color_spectrum=self.images.color_spectrum)
            self.histogram_functions.createHistogram(image,image_path=path)
            self.timing_results.append(time.time() - start_time)
        self.histogram_functions.averageHistogramsByType()
        self.histogram_functions.plotAveragedHistogramsByType()

    def k_means_segmentation(self,image):
        return self.segmentation.k_means_segmentation(image)

    def histogram_thresholding_segmentation(self,image):
        bin_values, bins = self.histogram_functions.createHistogram(image)
        return self.segmentation.histogram_thresholding_segmentation(image,bin_values,bins)

    def edge_detection(self,image,detection_type):
        return self.edges.edge_detection(image,detection_type=detection_type,threshold=self.edges.edge_detection_threshold)

    def edge_erosion(self,image):
        return self.edges.edge_erosion(image,num_layers=self.edges.num_erosion_layers,structuring_element=self.filters.edge_erosion_element)

    def edge_dilation(self,image):
        return self.edges.edge_dilation(image,num_layers=self.edges.num_dilation_layers,structuring_element=self.filters.edge_dilation_element)


    def parallelModel(self, path, function_list):
        """
        Method for performing image operations which can be parallelized.

        Parameters:
        -----------
            path(str) : the path of the image
            kwargs(dict) : the requested functions to be run from the .env file

        Returns:
        --------
            processing_time(float) : the processing time for the operation
            equalization_msqe(float) : the msqe for the quantization
        """

        current_process = multiprocessing.Process().name
        start_time = time.time()

        # get channel defined in .env file
        image = self.images.getImage(path,color_spectrum=self.images.color_spectrum)

        # apply all requested functions
        for i in function_list:
            # If k-means then it needs to be color image
            if i in ['10']:
                # get image again if it is not rgb (K-means must be done in color)
                if len(image.shape) != 3:
                    image = self.images.getImage(path,color_spectrum='rgb')
                image = self.function_dictionary[i](image)
            # Could be one of three edge detection functions
            elif i in ['14','15','16']:
                # if k-means was the previous function then image will be 3d and
                # it needs to be converted to greyscale
                if len(image.shape) == 3:
                    image = self.images.rgbToGrayscale(image)
                mapping = {'14':'sobel','15':'improved_sobel','16':'prewitt'}
                image = self.function_dictionary['14'](image,mapping[i])
                #self.images.showGrayscaleImages([image], num_rows=1, num_cols=1)
            else:
                # if k-means was the previous function then image will be 3d and
                # it needs to be converted to greyscale
                if len(image.shape) == 3:
                    image = self.images.rgbToGrayscale(image)
                image = self.function_dictionary[i](image)
                #self.images.showGrayscaleImages([image], num_rows=1, num_cols=1)

        if 'image' in locals():
            self.images.saveImage(image,path) # save image as grayscale

        print(f"{current_process} : done")

        # return the image processing time and equalization_msqe
        return [time.time() - start_time, self.equalization_msqe]


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
            print('Averaging histograms and exiting. Do not include this function if you wish to run any others!')
            self.function_dictionary['9']()
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

    start_time = time.time()

    composite.run_batch_mode()

    print("\n--- Batch Processing Time: %s seconds ---" % (time.time() - start_time))
    
    average_processing_time = sum(composite.timing_results)/len(composite.timing_results)
    print("--- Processing Time Per Image: %s seconds ---\n" % (average_processing_time))
    
    if composite.msqe_results:
        average_msqe = sum(composite.msqe_results)/len(composite.msqe_results)
        print("--- Average MSQE: %s ---\n" % (average_msqe))  

    # display up to four images at once
    #composite.images.showGrayscaleImages([salt_pepper_noise_image, filtered_sp_image], num_rows=1, num_cols=2)

    # Create and plot up to four histograms from a list of images
    #composite.histogram_functions.createAndPlotHistograms([grey_channel, salt_pepper_noise_image, filtered_sp_image], num_rows=2, num_cols=2)
