from dotenv import load_dotenv
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
        self.filters = None
        self.point_operations = None
        self.images = None
        self.noise_functions = None
        self.histogram_functions = None


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

    """THESE OPERATIONS WILL BE IN FOR LOOP PER IMAGE"""
    # break single image into channels
    red_channel, green_channel, blue_channel, grey_channel = composite.images.rgbToSingleChannels(composite.images.imagepaths[0])

    # use different filter operations on image
    box_image = composite.point_operations.smooth2dImage(grey_channel, composite.filters.box_filter['filter']) # use averaging (smoothing) of a grayscale image
    print("done box")
    median_image = composite.point_operations.medianOf2dImage(grey_channel, composite.filters.median_filter['filter']) # use median funcion on a grayscale image
    print("done median")
    gaussian_image = composite.point_operations.smooth2dImage(grey_channel, composite.filters.gaussian_filter['filter']) # use gaussian for averaging (smoothing) of a grayscale image
    print("done gaussian")
    difference_image = composite.point_operations.difference2dImage(grey_channel, composite.filters.laplacian_filter['filter']) # use difference of a grayscale image
    print("done difference")

    # use noise functions on grayscale image
    salt_pepper_noise_image = composite.noise_functions.addSaltAndPepperNoise(grey_channel)
    gaussian_noise_image = composite.noise_functions.addGaussianNoise(grey_channel)

    # filter salt and pepper image with median filter
    filtered_sp_image = composite.point_operations.medianOf2dImage(salt_pepper_noise_image, composite.filters.median_filter['filter'])

    # show images
    composite.images.showGrayscaleImages([salt_pepper_noise_image, filtered_sp_image], num_rows=1, num_cols=2)

    """
    #bin_values, bins = createHistogram(grey_channel)
    #plotHistogram(bin_values, bins)

    #plt.imsave('cell_images_original\cyl_cells\cyl01_modified.BMP', grey_channel, cmap='gray', vmin=0, vmax=255) #Save back grayscale image
     """
