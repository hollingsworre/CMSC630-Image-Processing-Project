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

        parallelModel(self, path, kwargs)
            Method for performing image operations which can be parallelized.

        run_batch_mode(self, **kwargs)
            Batch mode function. All kwargs arguments from the .env file should be converted to lower case
            before being passed into this function. If user is requesting an averaged histogram operation then
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


    def parallelModel(self, path, kwargs):
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

        equalization_msqe = 0
        current_process = multiprocessing.Process().name
        start_time = time.time()

        # get channel defined in .env file
        requested_channel = self.images.getImage(path,color_spectrum=self.images.color_spectrum)

        if kwargs['salt_pepper_noise'] == 'true':
            altered_image = self.noise_functions.addSaltAndPepperNoise(requested_channel)
        elif kwargs['gaussian_noise'] == 'true':
            altered_image = self.noise_functions.addGaussianNoise(requested_channel)
        elif kwargs['equalization'] == 'true':
            altered_image = self.histogram_functions.histogramEqualization(requested_channel)
        elif kwargs['quantization'] == 'true':
            altered_image = self.images.quantizeImage(requested_channel)
            decompressed_image = self.images.decompressImage(altered_image)
            equalization_msqe = self.images.quantizationError(requested_channel, decompressed_image)
        elif kwargs['box_smoothing'] == 'true':
            altered_image = self.point_operations.smooth2dImage(requested_channel, self.filters.box_filter['filter'])
            print(f"{current_process} : done box")
        elif kwargs['gaussian_smoothing'] == 'true':
            altered_image = self.point_operations.smooth2dImage(requested_channel, self.filters.gaussian_filter['filter'])
            print(f"{current_process} : done gaussian")
        elif kwargs['laplacian_diff'] == 'true':
            altered_image = self.point_operations.difference2dImage(requested_channel, self.filters.laplacian_filter['filter'])
            print(f"{current_process} : done difference")
        elif kwargs['median_smoothing'] == 'true':
            altered_image = self.point_operations.medianOf2dImage(requested_channel, self.filters.median_filter['filter'])
            print(f"{current_process} : done median")
        
        if 'altered_image' in locals():
            self.images.saveImage(altered_image,path) # save image as grayscale

        # return the image processing time and equalization_msqe
        return [time.time() - start_time, equalization_msqe]


    def run_batch_mode(self, **kwargs):
        """
        Batch mode function. All kwargs arguments from the .env file should be converted to lower case
        before being passed into this function. If user is requesting an averaged histogram operation then
        this will be performed synchronously by a single process. Otherwise a process pool equal to the 
        number of cpus on the machine will be spun up and the filepaths will be distributed to the processes
        for parallelization. Otherwise it takes a LONG time to process any significant number of images.

        Parameters:
        -----------
            **kwargs : the requested functions to be run from the .env file

        Returns:
        --------
            None
        """

        # If create_average_histograms, then just perform it synchronously for simplicity's sake.
        # Every image is looped through, histograms are totaled for each image type and then at
        # the end they are averaged and plotted. A lot easier than worrying about broadcasting between processes.
        if kwargs['create_average_histograms'] == 'true':
            for path in self.images.imagepaths:
                start_time = time.time()
                requested_channel = self.images.getImage(path,color_spectrum=self.images.color_spectrum)
                self.histogram_functions.createHistogram(requested_channel,image_path=path)
                self.timing_results.append(time.time() - start_time)
            self.histogram_functions.averageHistogramsByType()
            self.histogram_functions.plotAveragedHistogramsByType()
        # segment image synchronously
        elif kwargs['k_means'] == 'true':
            for path in self.images.imagepaths:
                start_time = time.time()
                image = self.images.getImage(path,color_spectrum='rgb')
                segmented_image = self.segmentation.k_means_segmentation(image)
                self.timing_results.append(time.time() - start_time)
                self.images.saveImage(segmented_image,path,cmap='rgb')
        # segment image synchronously
        elif kwargs['histogram_thresholding'] == 'true':
            for path in self.images.imagepaths:
                start_time = time.time()
                requested_channel = self.images.getImage(path,color_spectrum=self.images.color_spectrum)
                bin_values, bins = self.histogram_functions.createHistogram(requested_channel,image_path=path)
                segmented_image = self.segmentation.histogram_thresholding_segmentation(requested_channel,bin_values,bins)
                self.timing_results.append(time.time() - start_time)
                self.images.saveImage(segmented_image,path)
        elif kwargs['sobel_edge_detection'] == 'true':
            for path in self.images.imagepaths:
                start_time = time.time()
                grey_channel = self.images.getImage(path,color_spectrum=self.images.color_spectrum)
                # Smooth image with gaussian filter before doing edge detection
                smoothed_image = self.point_operations.smooth2dImage(grey_channel, self.filters.gaussian_filter['filter'])
                image_edges = self.edges.edge_detection(smoothed_image,detection_type='sobel',threshold=self.edges.edge_detection_threshold)
                #image_edges2 = self.edges.edge_detection(smoothed_image,detection_type='sobel',threshold=self.edges.edge_detection_threshold)
                #image_edges3 = self.edges.edge_detection(smoothed_image,detection_type='prewitt',threshold=self.edges.edge_detection_threshold)
                image_edges_erosion = self.edges.edge_erosion(image_edges,num_layers=1,structuring_element=self.filters.edge_erosion_element)
                image_edges_dilation = self.edges.edge_dilation(image_edges_erosion,num_layers=1,structuring_element=self.filters.edge_dilation_element)
                #image_edges_erosion = self.edges.edge_erosion(image_edges_dilation,num_layers=1)
                self.timing_results.append(time.time() - start_time)
                self.images.showGrayscaleImages([image_edges,image_edges_erosion,image_edges_dilation], num_rows=2, num_cols=2)
                #self.images.saveImage(image_edges_dilation,path)
        # else, if any other operation is requested then do it in parallel asynchronously for speed's sake
        else:
            # Create your process pool equal to the number of cpus detected on your machine
            pool = multiprocessing.Pool(os.cpu_count())
            # Use imagepaths iterable to dispatch paths to the process pool
            # Callback is used for storing the time of each image operation for final results at the end
            _ = [pool.apply_async(self.parallelModel, callback=self.save_data, args=(path, kwargs)) for path in self.images.imagepaths]
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

    # Pass in environment variables and convert the strings to lower for case insensitive comparisons
    composite.run_batch_mode(salt_pepper_noise=os.getenv('ADD_SALT_AND_PEPPER_NOISE').lower(),
                            gaussian_noise=os.getenv('ADD_GAUSSIAN_NOISE').lower(),
                            create_average_histograms=os.getenv('CREATE_AVERAGE_HISTOGRAMS').lower(),
                            equalization=os.getenv('RUN_HISTOGRAM_EQUALIZATION').lower(),
                            quantization=os.getenv('RUN_IMAGE_QUANTIZATION').lower(),
                            box_smoothing=os.getenv('RUN_LINEAR_BOX_SMOOTHING').lower(),
                            gaussian_smoothing=os.getenv('RUN_LINEAR_GAUSSIAN_SMOOTHING').lower(),
                            laplacian_diff=os.getenv('RUN_LINEAR_LAPLACIAN_DIFFERENCE').lower(),
                            median_smoothing=os.getenv('RUN_MEDIAN_SMOOTHING').lower(),
                            k_means=os.getenv('RUN_K_MEANS_SEGMENTATION').lower(),
                            histogram_thresholding=os.getenv('RUN_HISTOGRAM_SEGMENTATION').lower(),
                            sobel_edge_detection = os.getenv('RUN_SOBEL_EDGE_DETECTION').lower())

    
    print("\n--- Batch Processing Time: %s seconds ---" % (time.time() - start_time))
    
    average_processing_time = sum(composite.timing_results)/len(composite.timing_results)
    print("--- Processing Time Per Image: %s seconds ---\n" % (average_processing_time))
    
    if composite.msqe_results:
        average_msqe = sum(composite.msqe_results)/len(composite.msqe_results)
        print("--- Average MSQE: %s ---\n" % (average_msqe))

    function_list = os.getenv('FUNCTION_LIST')

    # display up to four images at once
    #composite.images.showGrayscaleImages([salt_pepper_noise_image, filtered_sp_image], num_rows=1, num_cols=2)

    # Create and plot up to four histograms from a list of images
    #composite.histogram_functions.createAndPlotHistograms([grey_channel, salt_pepper_noise_image, filtered_sp_image], num_rows=2, num_cols=2)
