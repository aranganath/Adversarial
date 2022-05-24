import numpy as np
from sklearn.svm import SVC
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import utils

class LayerwiseClustering():
    """
    """
    def __init__(self, model, dim_reducer=None, SVC_C=1):
        super().__init__()
        self.model = model
        self.SVC_C = SVC_C
        self.dim_reducer = dim_reducer
        
    def plotSampleInClusters(self, data, labels, sample):
        # Get model prediction probabilities for passed-in sample
        img = torch.Tensor(sample)
        img = img.unsqueeze(axis=0)
        with torch.no_grad():
            logps = self.model(img)

        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        print("Predicted Class =", probab.index(max(probab)))
        # Use probabilities to view-classify sample
        utils.view_classify(img.squeeze(), ps)
        
        # Add passed-in sample to the dataset
        sample_idx = data.shape[0]
        data = np.append(data, np.expand_dims(sample, axis=0), axis=0)
        
        # Get layerwise model outputs on all passed-in data
        layerwise_outputs = self.getLayerwiseOutputs(data)
        
        # For each layer whose output was saved, plot a TSNE embedding of the outputs with data separated by class and the passed-in
        # sample as a distinct point
        for i, outputs in enumerate(layerwise_outputs):
            x = outputs
            if (x.ndim > 2): # Flatten data
                x = x.reshape(x.shape[0], -1)
            if (x.shape[1] > 500 and self.dim_reducer != None): # Reduce dimensionality of data
                x = self.dim_reducer.fit_transform(x)
            tsne_output = TSNE(n_components=2, perplexity = 50, n_iter = 2000, learning_rate = 200.0, init='random').fit_transform(x)

            # For each unique class of the passed-in data, plot the TSNE embedding of that data
            classes = np.unique(labels)
            plt.figure(figsize=(16,10))
            for val in classes:
                class_indices = np.where(labels == val)
                plt.scatter(tsne_output[class_indices,0], tsne_output[class_indices,1], label=str(val))
            # Plot the sample on the same graph
            plt.scatter(tsne_output[sample_idx,0], tsne_output[sample_idx,1], label="Sample", c="#000000")
            
            # Label plot appropriately
            if (i == 0):
                plt.title("Inputs")
            elif(i == (len(layerwise_outputs) - 1)):
                plt.title("Outputs")
            else:
                plt.title("Output layer %d" % i)
            plt.legend()
            plt.show()

    def fit(self, X, y):
        layerwise_outputs = self.getLayerwiseOutputs(X)
                    
        self.feature_clusters = [SVC(kernel='rbf', C=self.SVC_C, random_state=42, max_iter = 1e5, decision_function_shape='ovr').fit(
                                     x, y) for x in layerwise_outputs]
        
    def predict(self, X):
        layerwise_outputs = self.getLayerwiseOutputs(X)
                    
        return [self.feature_clusters[i].predict(layerwise_outputs[i]) for i in range(len(self.feature_clusters))]
    
    # Modify to put model and tensors on GPU if it's available, then take model off GPU at the end of the method
    def getLayerwiseOutputs(self, X):
        X_tensor = torch.utils.data.TensorDataset(torch.Tensor(X))
        X_loader = torch.utils.data.DataLoader(X_tensor, batch_size=128, shuffle=False)
        
        layerwise_outputs = []
        
        for x in X_loader:
            x = x[0]
            batch_outputs = self.forward(x)
            if len(layerwise_outputs) == 0:
                layerwise_outputs = batch_outputs
            else:
                for i in range(len(layerwise_outputs)):
                    layerwise_outputs[i] = np.append(layerwise_outputs[i], batch_outputs[i], axis=0)
        
        return layerwise_outputs

    def forward(self, x):
        """ Returns a list of inputs to each convolutional and linear layer for the batch passed in as x.
        """
        batch_outputs = []
        
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass of model, saving inputs to each convolutional or linear layer
            for m in self.model.children():
                # If module is a list of layers, iterate through list
                if isinstance(m, nn.Sequential):
                    for l in m:
                        if isinstance(l, nn.Conv2d) or isinstance(l, nn.Linear):
                            batch_outputs.append(x.numpy())
                        x = l(x)
                else:
                    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                        batch_outputs.append(x.numpy())
                    x = m(x)
        
        batch_outputs.append(x.numpy()) # Append model output
        
        self.model.train()
        
        return batch_outputs