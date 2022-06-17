import numpy as np
from scipy import fftpack

class DCTDimReducer():
    def __init__(self, input_image_shape, coef_size_reduction=0, inverse=False):
        self.coef_size_reduction = coef_size_reduction
        self.input_image_shape = input_image_shape
        self.inverse = inverse
        
    def fit_transform(self, X, y=None):
        """
            X = dataset to apply discrete cosine transform to
            y is not used
        """
        shape = self.input_image_shape
        reduction = self.coef_size_reduction
        
        if self.inverse and reduction > 0:
            # Allocate space for reshaped data
            data = np.zeros((X.shape[0], shape[0], shape[1]), dtype=float)
            
            # For each sample, we assume that a square of size (reductionxreduction) was removed on the forward pass of the dct
            # In order to perform the idct, we need to get the original shape back, appending zeros where elements were removed
            for i in range(X.shape[0]):
                sample = X[i]
                length_top_portion = (shape[0] - reduction)*shape[1]
                
                # Reshape 'top' portion of the flattened out vector, comprised of rows in the original image that didn't have
                # any elements taken off of the end of them
                top = sample[:length_top_portion].reshape([(shape[0] - reduction), shape[1]])
                
                # Reshape the bottom left portion, where we assume that each row is missing <reduction> number of elements
                bottom_left = sample[length_top_portion:].reshape([reduction, (shape[1] - reduction)])
                # Create bottom right portion, by creating a reductionxreduction square of zeros
                bottom_right = np.zeros([reduction,reduction], dtype=float)
                bottom = np.concatenate((bottom_left, bottom_right), axis=1)
                
                # Put all elements together into the appropriate shape and add to data array
                data[i] = np.concatenate((top, bottom), axis=0)
                
            # Allocate space for coefficient vectors
            coefficients = np.zeros([X.shape[0], shape[0]*shape[1]], dtype=float)
        elif reduction > 0:
            # Get the indices of the flattened-out versions of images to delete in order to eliminate square of size 
            # reductionxreduction in the bottom-right corner of the image
            indices = np.zeros([2,reduction**2], dtype=np.uint8)

            for i in range(0,reduction):
                indices[0,i*reduction:(i+1)*reduction] = shape[0] - reduction + i
                for j in range(0,reduction):
                    indices[1,i*reduction + j] = shape[1] - reduction + j
            
            # Allocate space for coefficient vectors, accounting for dim reduction
            coefficients = np.zeros([X.shape[0], shape[0]*shape[1] - reduction**2], dtype=float)
            
            # Reshape data to 2d images for discrete cosine transform
            data = X.reshape(X.shape[0], shape[0], shape[1])
        else:
            # Allocate space for coefficient vectors
            coefficients = np.zeros([X.shape[0], shape[0]*shape[1]], dtype=float)
            
            # Reshape data to 2d images for discrete cosine transform
            data = X.reshape(X.shape[0], shape[0], shape[1])
        
        for i in range(data.shape[0]):
            coef = data[i]
            for dim in range(data[0].ndim):
                if self.inverse:
                    coef = fftpack.idct(coef, axis=dim)
                else:
                    coef = fftpack.dct(coef, axis=dim)
            
            coef = coef.flatten()
            
            if reduction > 0 and not self.inverse:
                coef = np.delete(coef, np.ravel_multi_index(indices, shape))
            coefficients[i] = coef
        
        return coefficients