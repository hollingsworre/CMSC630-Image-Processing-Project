import numpy as np
import random
import os


class Noise:
    """
    Component class for all noise functions to be added to the image.

    Attributes
    ----------

    gaussian_noise_strength(int) : Strength of noise to apply to gaussian function. Loaded from .env file.
    num_salt_pixels(int) : Number of salt pixels to add as noise. Loaded from .env file.
    num_pepper_pixels(int) : Number of pepper pixels to add as noise. Loaded from .env file.

    Methods
    -------

    addSaltAndPepperNoise(self,image)
        Randomly adds user specified number of salt and pepper pixels to an image

    addGaussianNoise(self,image)
        Adds gaussian noise to an image by randomly sampling values from a normal distribution with
        the strength specified from the .env file. These random samples are then added to the original
        image to create a noisy image.

    """

    def __init__(self):
        self.gaussian_noise_strength = int(os.getenv('GAUSSIAN_NOISE_STRENGTH'))
        self.num_salt_pixels = int(os.getenv('NUM_SALT_PIXELS'))
        self.num_pepper_pixels = int(os.getenv('NUM_PEPPER_PIXELS'))
    

    def addSaltAndPepperNoise(self,image):
        """
        Randomly adds user specified number of salt and pepper pixels to an image

        Parameters:
            image(numpy array): the image to add noise to

        Returns:
            image_noisy (numpy array): The altered image
        """
        
        image_noisy= np.copy(image) # make copy of image

        # Getting the dimensions of the image
        row , col = image_noisy.shape

        if (self.num_salt_pixels < 0 or self.num_salt_pixels > image_noisy.size):
            print("Randomly setting number of salt pixels")
            self.num_salt_pixels = random.randint(0, image_noisy.size)
        if (self.num_pepper_pixels < 0 or self.num_pepper_pixels > image_noisy.size):
            print("Randomly setting number of pepper pixels")
            self.num_pepper_pixels = random.randint(0, image_noisy.size)
        
        # add user specified number of salt pixels at random positions
        for _ in range(self.num_salt_pixels):       
            # Pick a random y coordinate
            y_coord=random.randint(0, row - 1)         
            # Pick a random x coordinate
            x_coord=random.randint(0, col - 1)         
            # Color that pixel to white
            image_noisy[y_coord][x_coord] = 255
    
        # add user specified number of pepper pixels at random positions
        for _ in range(self.num_pepper_pixels):       
            # Pick a random y coordinate
            y_coord=random.randint(0, row - 1)         
            # Pick a random x coordinate
            x_coord=random.randint(0, col - 1)         
            # Color that pixel to black
            image_noisy[y_coord][x_coord] = 0
            
        return image_noisy # return the altered image


    def addGaussianNoise(self,image):
        """
        Adds gaussian noise to an image by randomly sampling values from a normal distribution with
        the strength specified from the .env file. These random samples are then added to the original
        image to create a noisy image.

        Parameters:
            image(numpy array): The image to add noise to.

        Returns:
            image_noisy: The corrupted image which is a sum of the original image and the random gaussian
            noisy distribution.
        """

        image_noisy= np.empty_like(image) # make copy of image
        
        if self.gaussian_noise_strength <= 0: 
            print("Gaussian strength of noise needs to be greater than zero. Setting strength to 1")
            self.gaussian_noise_strength = 1

        noise = np.random.normal(0, self.gaussian_noise_strength, size = image.shape)
        image_noisy = image + noise

        # Normalize image between 0 and 255
        image_noisy = 255*((image_noisy-np.min(image_noisy))/(np.max(image_noisy)-np.min(image_noisy)))

        return image_noisy 
