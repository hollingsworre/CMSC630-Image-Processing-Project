from dotenv import load_dotenv
import os
from components.filters import Filters
from components.point_operations import ImagePointOperations
from components.images import Images
from components.noise import Noise
from components.histogram import Histogram


class Main:
    """
    Composite class made up of all components.
    """

    def __init__(self):        
        # component classes that can be added
        self.filters = None
        self.point_operations = None
        self.images = None
        self.noise_functions = None
        self.histogram_functions = None

    def run_batch_mode(self, **kwargs):
        """
        Batch mode function. All kwargs arguments from the .env file should be converted to lower case
        before being passed into this function.
        """

        # Go through each imagepath, load the image and perform the requested operation from the .env file
        for path in self.images.imagepaths:
            # get channel defined in .env file
            requested_channel = self.images.rgbToSingleChannels(path)

            if kwargs['salt_pepper_noise'] == 'true':
                altered_image = self.noise_functions.addSaltAndPepperNoise(requested_channel)
            elif kwargs['gaussian_noise'] == 'true':
                altered_image = self.noise_functions.addGaussianNoise(requested_channel)
            elif kwargs['create_average_histograms'] == 'true':
                self.histogram_functions.createHistogram(requested_channel,image_path=path)
            elif kwargs['equalization'] == 'true':
                altered_image = self.histogram_functions.histogramEqualization(requested_channel)
            elif kwargs['quantization'] == 'true':
                altered_image = self.images.quantizeImage(requested_channel)
                decompressed_image = self.images.decompressImage(altered_image)
                equalization_msqe = self.images.quantizationError(requested_channel, decompressed_image)
                print(f'MSQE = {equalization_msqe}')
            elif kwargs['box_smoothing'] == 'true':
                altered_image = self.point_operations.smooth2dImage(requested_channel, self.filters.box_filter['filter'])
                print("done box")
            elif kwargs['gaussian_smoothing'] == 'true':
                altered_image = self.point_operations.smooth2dImage(requested_channel, self.filters.gaussian_filter['filter'])
                print("done gaussian")
            elif kwargs['laplacian_diff'] == 'true':
                altered_image = self.point_operations.difference2dImage(requested_channel, self.filters.laplacian_filter['filter'])
                print("done difference")
            elif kwargs['median_smoothing'] == 'true':
                altered_image = self.point_operations.medianOf2dImage(requested_channel, self.filters.median_filter['filter'])
                print("done median")
            
            if 'altered_image' in locals():
                self.images.saveImage(altered_image,path)

        if kwargs['create_average_histograms'] == 'true':
            self.histogram_functions.averageHistogramsByType()
            self.histogram_functions.plotAveragedHistogramsByType()


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

    # Pass in environment variables and convert the strings to lower for case insensitive comparisons
    composite.run_batch_mode(salt_pepper_noise=os.getenv('ADD_SALT_AND_PEPPER_NOISE').lower(),
                            gaussian_noise=os.getenv('ADD_GAUSSIAN_NOISE').lower(),
                            create_average_histograms=os.getenv('CREATE_AVERAGE_HISTOGRAMS').lower(),
                            equalization=os.getenv('RUN_HISTOGRAM_EQUALIZATION').lower(),
                            quantization=os.getenv('RUN_IMAGE_QUANTIZATION').lower(),
                            box_smoothing=os.getenv('RUN_LINEAR_BOX_SMOOTHING').lower(),
                            gaussian_smoothing=os.getenv('RUN_LINEAR_GAUSSIAN_SMOOTHING').lower(),
                            laplacian_diff=os.getenv('RUN_LINEAR_LAPLACIAN_DIFFERENCE').lower(),
                            median_smoothing=os.getenv('RUN_MEDIAN_SMOOTHING').lower())

    # display up to four images at once
    #composite.images.showGrayscaleImages([salt_pepper_noise_image, filtered_sp_image], num_rows=1, num_cols=2)

    # Create and plot up to four histograms from a list of images
    #composite.histogram_functions.createAndPlotHistograms([grey_channel, salt_pepper_noise_image, filtered_sp_image], num_rows=2, num_cols=2)
