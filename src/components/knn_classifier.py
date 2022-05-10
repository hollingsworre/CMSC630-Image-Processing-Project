import numpy as np
import pandas as pd
import os
import copy
from math import sqrt


class KNN_Classifer:
    """
    Component class for KNN training and classification
    """

    def __init__(self):
        self.num_neighbors = int(os.getenv('NUM_KNN_NEIGHBORS'))
        self.training_dataset_path = os.getenv('TRAINING_DATASET_PATH')
        self.num_k_folds = int(os.getenv('NUM_K_FOLDS'))        
        self.training_dataset = self.load_csv_dataset(self.training_dataset_path)        


    def load_csv_dataset(self,csv_path,random_state=1):
        """
        Loads .csv without the header into panda dataframe, shuffles it, converts it to numpy matrix.
        """
        if os.path.exists(csv_path):
            dataframe = pd.read_csv(csv_path, header=0) # load csv into pandas dataframe
            shuffled = dataframe.sample(frac=1, random_state=random_state).reset_index() # shuffle dataframe
            
            # create a numpy array with the numeric values for input into classifier
            numpy_array = shuffled.to_numpy()        
            return numpy_array
        else:
            return None


    def run_k_fold_knn(self):
        """Runs k-fold knn cross validation"""

        average_accuracy_total = 0       

        # split training dataset into k chunks
        k_fold_chunks = np.array_split(self.training_dataset,self.num_k_folds)

        # Use all sets for validation against other datasets
        # For a total of k validations
        for i in range(self.num_k_folds):
            num_correct_predictions = 0
            # Get your validations set for this round
            validation_set = k_fold_chunks[i]

            # Get your training dataset for this round
            training_dataset = copy.deepcopy(np.array(k_fold_chunks,dtype=object))
            training_dataset = np.delete(training_dataset,[i])
            training_dataset = np.concatenate(training_dataset)

            # for each sample in the validation set get it's nearest neighbors
            for validation_sample in validation_set:
                prediction = self.predict_class(training_dataset, validation_sample)
                # if prediction is correct then count it for accuracy calcs
                if prediction == validation_sample[len(validation_sample)-1]:
                    num_correct_predictions = num_correct_predictions + 1

            accuracy = (num_correct_predictions/len(validation_set)) * 100
            average_accuracy_total = average_accuracy_total + accuracy
            print(f'Accuracy for {self.num_neighbors} neighbors on validation set {i}:  {accuracy:.2f}%')

        print(f'\nAverage accuracy for {self.num_neighbors} neighbors on {self.num_k_folds} folds:  {(average_accuracy_total/self.num_k_folds):.2f}%')


    def predict_class(self, training_dataset, validation_sample):
        """Make knn prediction"""
        distances = []
        for training_row in training_dataset:
            distance = self.euclidean_distance(training_row, validation_sample)
            distances.append((distance,training_row[len(training_row)-1])) # append distance and label of training row

        distances.sort(key=lambda tup: tup[0]) # sort by distance

        nearest_neighbors = []
        for i in range(self.num_neighbors):
            nearest_neighbors.append(distances[i][1]) # compile specified number of nearest neighbors into a list

        nearest_list = set(nearest_neighbors)
        prediction = max(nearest_list, key=nearest_neighbors.count) # in case of ties picks the first max encountered in the list

        return prediction
        

    def euclidean_distance(self, training_row, validation_row):
        """Calculate distance between two row vectors. Ignores the class label at the end."""
        distance = 0.0
        for i in range(len(training_row)-1):
            distance += (training_row[i] - validation_row[i])**2
        return sqrt(distance)
