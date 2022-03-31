from xmlrpc.client import Boolean
import numpy as np
import os


class Filters:
    """
    Component class for all filters/structuring elements to be worked with. When this class is instantiated the
    elements will be parsed from the .env file and separated into row and column vectors if possible.
    If they can not be separated then the row_vector and column_vector values of the filter
    dicts will be None

    Attributes
    ----------

    laplacian_filter : a laplacian filter defined in .env in form of::

        dict:{
                'filter': numpy matrix,
                'column_vector': None(if not separable) or numpy array(if separable),
                'row_vector': None(if not separable) or numpy array(if separable)
            }

    gaussian_filter : a gaussian filter defined in .env in form of::

        dict:{
                'filter': numpy matrix,
                'column_vector': None(if not separable) or numpy array(if separable),
                'row_vector': None(if not separable) or numpy array(if separable)
            }

    box_filter : a box filter defined in .env in form of::

        dict:{
                'filter': numpy matrix,
                'column_vector': None(if not separable) or numpy array(if separable),
                'row_vector': None(if not separable) or numpy array(if separable)
            }

    median_filter : a median filter defined in .env in form of::

        dict:{
                'filter': numpy matrix,
                'column_vector': None(if not separable) or numpy array(if separable),
                'row_vector': None(if not separable) or numpy array(if separable)
            }

    edge_dilation_element : structuring element used for edge dilation

    edge_erosion_element : structuring element used for edge erosion

    Methods
    -------

        getFilterMatrix(self,filter)
            Gets filter matrix from that which is defined in the .env file and converts the string 
            values into an appropriate numpy array.

        separateFilter(self,filter)
            Separates a filter of rank 1 (Box or Median usually) into row and column vectors which
            when multiplied back together will recompose the original filter. This is useful for 
            image processing convolution.
    """

    def __init__(self):
        # get all filters and place them into a dict under the key 'filter'
        self.laplacian_filter = self.separateFilter({'filter':self.getFilterMatrix(os.getenv('LINEAR_LAPLACIAN_DIFFERENCE_FILTER').splitlines()),
                                                    'column_vector':None,
                                                    'row_vector':None}) #get laplacian filter
        self.gaussian_filter = self.separateFilter({'filter':self.getFilterMatrix(os.getenv('LINEAR_GAUSSIAN_SMOOTHING_FILTER').splitlines()),
                                                    'column_vector':None,
                                                    'row_vector':None}) #get gaussian filter
        self.box_filter = self.separateFilter({'filter':self.getFilterMatrix(os.getenv('LINEAR_BOX_SMOOTHING_FILTER').splitlines()),
                                                'column_vector':None,
                                                'row_vector':None}) #get box filter
        self.median_filter = self.separateFilter({'filter':self.getFilterMatrix(os.getenv('MEDIAN_FILTER').splitlines()),
                                                    'column_vector':None,
                                                    'row_vector':None}) #get median filter
        self.edge_dilation_element = self.getFilterMatrix(os.getenv('EDGE_DILATION_STRUCTURING_ELEMENT').splitlines(),type=Boolean)
        self.edge_erosion_element = self.getFilterMatrix(os.getenv('EDGE_EROSION_STRUCTURING_ELEMENT').splitlines(),type=Boolean)


    def getFilterMatrix(self,filter,type=int):
        """
        Gets filter matrix from that which is defined in the .env file and converts the string 
        values into an appropriate numpy array.

        Parameters:
        -----------
            filter(list): list representing the rows of the filter and gotten from the .env file
            type : data type to cast the matrix to. Usually int for filters and Boolean for structuring elements

        Returns:
        --------
            array: filter as a 2D numpy array
        """
        
        array = []
        # Each list item (other than the first) represents a row of the filter matrix    
        for row in range(1, len(filter)):
            filter_list = filter[row].split(',')
            array.append(list(map(type,filter_list))) #build 2D array

        # TODO: width and height of filter should both be odd values (probably should check for this)
        return np.asarray(array) # return as numpy array


    def separateFilter(self,filter):
        """
        Separates a filter of rank 1 (Box or Median usually) into row and column vectors which
        when multiplied back together will recompose the original filter. This is useful for 
        image processing convolution.

        Parameters:
        -----------
            filter(dict): dict element with filter to be separated listed under the 'filter' key

        Returns:
        --------
            filter(dict): if the filter can be separated (rank 1) then the separated row and column vectors
            will be listed under the keys 'column_vector' and 'row_vector'. Else these keys will not be present.
        """
        # decompose filter into SVD row and column parts
        # if matrix rank is 1 (Box or Median) then it can be easily decomposed into column and row matrices by this method
        if (np.linalg.matrix_rank(filter['filter']) == 1):
            # decompose filter into column and row vectors
            U, s, V = np.linalg.svd(filter['filter'])
            U_column = U[:,0]*np.sqrt(s[0]) # column matrix (only need weight of first element in s b/c filter is rank 1)
            U_column = U_column.reshape(U_column.shape[0],1)
            V_row = np.sqrt(s[0])*V[0,:]  # row matrix (only need weight of first element in s b/c filter is rank 1)
            V_row = V_row.reshape(1,V_row.shape[0])
            filter['column_vector'] = U_column # place column vector into filter dict
            filter['row_vector'] = V_row    #place row_vector into filter dict
            
        return filter #return the filter
 