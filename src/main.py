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
        # Go through each imagepath, load the image and perform the requested operations
        for path in self.images.imagepaths:
            red_channel, green_channel, blue_channel, grey_channel = self.images.rgbToSingleChannels(path)
            if kwargs['salt_pepper_noise'] == 'TRUE':
                altered_image = self.noise_functions.addSaltAndPepperNoise(grey_channel)
            if kwargs['gaussian_noise'] == 'TRUE':
                altered_image = self.noise_functions.addGaussianNoise(grey_channel)
            if kwargs['create_average_histograms'] == 'TRUE':
                self.histogram_functions.createHistogram(grey_channel,image_path=path)
            if kwargs['equalization'] == 'TRUE':
                altered_image = self.histogram_functions.histogramEqualization(grey_channel)
            if kwargs['quantization'] == 'TRUE':
                altered_image = self.images.quantizeImage(grey_channel)
                decompressed_image = self.images.decompressImage(altered_image)
                equalization_msqe = self.images.quantizationError(grey_channel, decompressed_image)
                print(f'MSQE = {equalization_msqe}')
            if kwargs['box_smoothing'] == 'TRUE':
                altered_image = self.point_operations.smooth2dImage(grey_channel, self.filters.box_filter['filter'])
                print("done box")
            if kwargs['gaussian_smoothing'] == 'TRUE':
                altered_image = self.point_operations.smooth2dImage(grey_channel, self.filters.gaussian_filter['filter'])
                print("done gaussian")
            if kwargs['laplacian_diff'] == 'TRUE':
                altered_image = self.point_operations.difference2dImage(grey_channel, self.filters.laplacian_filter['filter'])
                print("done difference")
            if kwargs['median_smoothing'] == 'TRUE':
                altered_image = self.point_operations.medianOf2dImage(grey_channel, self.filters.median_filter['filter'])
                print("done median")
            
            if 'altered_image' in locals():
                self.images.saveImage(altered_image,path)

        if kwargs['create_average_histograms'] == 'TRUE':
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

    composite.run_batch_mode(salt_pepper_noise=os.getenv('ADD_SALT_AND_PEPPER_NOISE'),
                            gaussian_noise=os.getenv('ADD_GAUSSIAN_NOISE'),
                            create_average_histograms=os.getenv('CREATE_AVERAGE_HISTOGRAMS'),
                            equalization=os.getenv('RUN_HISTOGRAM_EQUALIZATION'),
                            quantization=os.getenv('RUN_IMAGE_QUANTIZATION'),
                            box_smoothing=os.getenv('RUN_LINEAR_BOX_SMOOTHING'),
                            gaussian_smoothing=os.getenv('RUN_LINEAR_GAUSSIAN_SMOOTHING'),
                            laplacian_diff=os.getenv('RUN_LINEAR_LAPLACIAN_DIFFERENCE'),
                            median_smoothing=os.getenv('RUN_MEDIAN_SMOOTHING'))


    """ 
    # show images
    composite.images.showGrayscaleImages([salt_pepper_noise_image, filtered_sp_image], num_rows=1, num_cols=2)

    # Create and plot histograms from a list of images
    composite.histogram_functions.createAndPlotHistograms([grey_channel, salt_pepper_noise_image, filtered_sp_image], num_rows=2, num_cols=2)

    """      

    """ 
    # compress and decompress images then show original vs decompressed image
    compressed_image = composite.images.quantizeImage(grey_channel)
    decompressed_image = composite.images.decompressImage(compressed_image)
    composite.histogram_functions.createAndPlotHistograms([grey_channel,decompressed_image],num_cols=2)
    composite.images.showGrayscaleImages([grey_channel, decompressed_image], num_rows=1, num_cols=2)
    equalization_msqe = composite.images.quantizationError(grey_channel, decompressed_image)    
    """