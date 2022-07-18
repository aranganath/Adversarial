import numpy as np
from scipy import fftpack
from time import time

class DCTDimReducer():
    def __init__(self, coef_size_reduction=0):
        self.coef_size_reduction = coef_size_reduction
        
    def fit_transform(self, X, return_flat=True):
        """
            X = dataset to apply discrete cosine transform to
            return_flat = bool indicating whether the coefficients of the dct transform should be returned
                without broadcasting back to the image's original shape
        """
        time0 = time()
        reduction = self.coef_size_reduction
        
        self.reshaped = False
        if X.shape[1] == 1 or X.shape[1] == 3:
            X = np.transpose(X, (0,2,3,1)) # Move channel dimension to end of image
            self.reshaped = True # Remember to move channel dimension to original place before returning
        self.input_shape = X.shape

        shape = self.input_shape
        
        # Setup for performing forward-dct and reducing dimensions
        if reduction > 0:
            # Create indices of the elements to be deleted in order to perform reduction
            # Since a square with <reduction> elements to a side is deleted, the indices vector
            #   will have 1 dimension of size 3 for the x, y, z coordinates of each element to be deleted,
            #   and one dimension of size <reduction>^2 for each element to be deleted

            # Generate x,y,z coordinates of elements to be deleted
            x, y, z = np.arange(shape[1]-reduction, shape[1]), np.arange(shape[2]-reduction, shape[2]), np.arange(0, shape[3])
            x, z = np.repeat(x, reduction), np.repeat(z, reduction**2)
            x, y = np.tile(x, shape[3]), np.tile(y, reduction*shape[3])

            indices = np.stack((x,y,z))

            # Allocate space for coefficient vectors, accounting for dim reduction
            coefficients = np.zeros([shape[0], shape[1]*shape[2]*shape[3] - reduction**2*shape[3]], dtype=float)
        
        # Setup for performing forward-dct and no dimensionality reduction was done
        else:
            # Allocate space for coefficient vectors
            coefficients = np.zeros([shape[0], shape[1]*shape[2]*shape[3]], dtype=float)

        # Actually doing dct, applying dct transform to each axis, going over the channel axis last
        for i in range(shape[0]): # Iterate over images along batch dimension
            coef = X[i]
            for dim in range(coef.ndim): # Iterate over each dimension of the current image
                coef = fftpack.dct(coef, axis=dim)

            coef = coef.flatten()

            # If we want to reduce dimension, delete elements at coordinates generated above
            if reduction > 0:
                coef = np.delete(coef, np.ravel_multi_index(indices, shape[1:]))

            coefficients[i] = coef

        self.transform_time = time() - time0
        if return_flat:
            return coefficients
        else:
            # If reduction was performed, the missing elements will be padded with 0s
            # if reduction was not performed, just reshapes the data to the original image dimensions
            coefficients = self.reshape_coefficients(coefficients, self.input_shape, reduction)
            return coefficients
            
    def inverse_fit_transform(self, X):
        """
            X = dataset to apply discrete cosine transform to
        """
        reduction = self.coef_size_reduction
        shape = self.input_shape
        
        # Setup for performing inverse dct and reduction was done on the forward-dct
        if reduction > 0 and X.ndim < 4:
            X = self.reshape_coefficients(X, self.input_shape, reduction)
                
        # Allocate space for coefficient vectors
        coefficients = np.zeros([shape[0], shape[1]*shape[2]*shape[3]], dtype=float)

        # Actually doing dct, applying dct transform to each axis, going over the channel axis last
        for i in range(shape[0]): # Iterate over images along batch dimension
            coef = X[i]
            for dim in range(coef.ndim): # Iterate over each dimension of the current image
                    coef = fftpack.idct(coef, axis=dim)

            coef = coef.flatten()

            coefficients[i] = coef

        # If reduction was performed, the missing elements will be padded with 0s
        # if reduction was not performed, just reshapes the data to the original image dimensions
        coefficients = self.reshape_coefficients(coefficients, self.input_shape, reduction)
        if self.reshaped: # Put image back into the shape that it was given in
            coefficients = np.transpose(coefficients, (0,3,1,2))
        return coefficients
    
    def reshape_coefficients(self, X, orig_shape, reduction):
        orig_length = 1
        for i in range(1,len(orig_shape)):
            orig_length = orig_length * orig_shape[i]

        if reduction <= 0 or X.shape[1] == orig_length:
            return X.reshape(orig_shape)

        # Allocate space for reshaped data
        data = np.zeros(orig_shape, dtype=float)

        # For each sample, we assume that a square of size (reductionxreduction) was removed on the forward pass of the dct
        # In order to perform the idct, we need to get the original shape back, appending zeros where elements were removed
        for i in range(orig_shape[0]):
            image = np.zeros((orig_shape[1:]), dtype=float)
            # Get the elements associated with the current channel according to the original shape
            # and reduction
            channel_data_length = orig_shape[1]*orig_shape[2] - reduction**2
            sample = X[i].reshape((channel_data_length,orig_shape[3]))
            for ch in range(orig_shape[3]): # For each channel

                length_top_portion = (orig_shape[1] - reduction)*orig_shape[2]

                # Reshape 'top' portion of the flattened out vector, comprised of rows in the original image that didn't have
                # any elements taken off of the end of them
                top = sample[:length_top_portion,ch].reshape([(orig_shape[1] - reduction), orig_shape[2]])

                # Reshape the bottom left portion, where we assume that each row is missing <reduction> number of elements
                bottom_left = sample[length_top_portion:,ch].reshape([reduction, (orig_shape[2] - reduction)])
                # Create bottom right portion, by creating a reductionxreduction square of zeros
                bottom_right = np.zeros([reduction,reduction], dtype=float)
                bottom = np.concatenate((bottom_left, bottom_right), axis=1)

                # Put all elements together into the appropriate shape and add to data array
                image[:,:,ch] = np.concatenate((top, bottom), axis=0)
            data[i] = image

        return data