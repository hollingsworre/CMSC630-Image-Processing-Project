from dotenv import load_dotenv
from filters import Filters
from point_operations import ImagePointOperations
from images import Images
from noise import Noise
from histogram import Histogram


class Main:
    """
    Composite class made up of all components.
    """

    def __init__(self):
        self.filters = Filters()
        self.point_operations = ImagePointOperations()
        self.images = Images()
        self.noise_functions = Noise()
        self.histogram_functions = Histogram()


if __name__ == "__main__":
    load_dotenv() # parse and load .env file into the environment
    main_component = Main()


    """ 

    column_matrix, row_matrix = separateFilter(median_filter)
    print(column_matrix)
    print(row_matrix)
    print(column_matrix@row_matrix) """
    
    


    """ imagepaths = getAllPathsInFolderByType() # get list of filepaths by file extension

    red_channel, green_channel, blue_channel, grey_channel = rgbToSingleChannels(imagepaths[0])

    box_image = smooth2dImage(grey_channel, box_filter) # use averaging (smoothing) of a grayscale image
    print("done box")
    gaussian_image = smooth2dImage(grey_channel, gaussian_filter) # use averaging (smoothing) of a grayscale image
    print("done gaussian") """
    """ difference_image = difference2dImage(grey_channel, laplacian_filter) # use difference of a grayscale image
    print("done difference") """
    """ median_image = medianOf2dImage(grey_channel, median_filter) # use difference of a grayscale image
    print("done median")

    #bin_values, bins = createHistogram(grey_channel)
    #plotHistogram(bin_values, bins)
    #noisy_image = addSaltAndPepperNoise(grey_channel, num_salt_pixels=1000, num_pepper_pixels=1000)
    #noisy_image = addGaussianNoise(grey_channel, std_dev=10)

    #plt.imsave('cell_images_original\cyl_cells\cyl01_modified.BMP', grey_channel, cmap='gray', vmin=0, vmax=255) #Save back grayscale image
    plt.subplot(2, 2, 1)
    plt.imshow(box_image, cmap='gray', vmin=0, vmax=255)
    plt.title('Box')
    plt.subplot(2, 2, 2)
    plt.imshow(median_image, cmap='gray', vmin=0, vmax=255)
    plt.title('Median')
    plt.subplot(2, 2, 3)
    plt.imshow(gaussian_image, cmap='gray', vmin=0, vmax=255)
    plt.title('Gaussian')
    plt.subplot(2, 2, 4)
    plt.imshow(grey_channel, cmap='gray', vmin=0, vmax=255)
    plt.title('Original')
    plt.show() """
