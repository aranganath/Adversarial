import numpy as np
from sklearn.svm import SVC
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import utils

class LayerwiseClustering():
    """ Creates an SVM with a RBF boundary around each class at each layer of the provided model.
        Can also be used to return the output of each linear or convolutional layer 'block' in the
        provided model.
    """
    def __init__(self, model, dim_reducer=None, SVC_C=1):
        super().__init__()
        self.model = model
        self.SVC_C = SVC_C
        self.dim_reducer = dim_reducer
        
    def plotClusters(self, data, labels, sample=None, rbf=False, only_in_and_out_layers=False):
        """ Plots a TSNE visualization of the output of each layer 'block' in the provided neural net.
            Also plots a provided sample in each layer's visualization, which is displayed distinctly 
            from each class of data provided.
        """
#         ###########################################
#         one_v_all_labels = np.zeros(labels.shape, dtype=np.uint8)
#         one_v_all_labels[np.where(labels == 0)[0]] = 1
#         ###########################################
        if sample != None:
            # Get model prediction probabilities for passed-in sample
            img = torch.Tensor(sample)
            img = img.unsqueeze(axis=0)
            with torch.no_grad():
                logps = self.model(img)

            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            print("Predicted Class =", probab.index(max(probab)))
            # Use probabilities to view-classify sample
            utils.view_classify(img.squeeze(), ps, img_title = "Adversarial Sample")
        
            # Add passed-in sample to the dataset
            sample_idx = data.shape[0]
            data = np.append(data, np.expand_dims(sample, axis=0), axis=0)
        
        # Get layerwise model outputs on all passed-in data
        layerwise_outputs = self.getLayerwiseOutputs(data)
        
        # For each layer whose output was saved, plot a TSNE embedding of the outputs with data separated by class and the passed-in
        # sample as a distinct point
        for i, outputs in enumerate(layerwise_outputs):
            if only_in_and_out_layers and (i == 0 or i == len(layerwise_outputs) - 1):
                x = outputs
                if (x.shape[1] > 500 and self.dim_reducer != None): # Reduce dimensionality of data
                    x = self.dim_reducer.fit_transform(x)
                x = x.reshape(x.shape[0], -1) # Flatten data
                tsne_output = TSNE(n_components=2, perplexity = 50, n_iter = 50000, learning_rate = 200.0, init='random').fit_transform(x)

                # For each unique class of the passed-in data, plot the TSNE embedding of that data
                classes = np.unique(labels)
                #colors=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'purple', 'black', 'gray', 'lightcoral']
                plt.figure(figsize=(16,10))
                for j, val in enumerate(classes):
                    class_indices = np.where(labels == val)
                    plt.scatter(tsne_output[class_indices,0], tsne_output[class_indices,1], label=str(val), 
                                #c=colors[j]
                               )
# #########################################################################
#                 # For each unique class of the passed-in data, plot the TSNE embedding of that data
#                 plt.figure(figsize=(16,10))
#                 for j in range(2):
#                     class_indices = np.where(one_v_all_labels == j)
#                     label = '0s' if j == 1 else 'Non 0s'
#                     plt.scatter(tsne_output[class_indices,0], tsne_output[class_indices,1], label=str(label))
# #########################################################################
                if sample != None:
                    # Plot the sample on the same graph
                    plt.scatter(tsne_output[sample_idx,0], tsne_output[sample_idx,1], label="Sample", c="#000000")

                if rbf:
                    ###############################################
                    clf = SVC(kernel='rbf', C=1, random_state=42, max_iter = 1e5).fit(tsne_output, labels)
                    
                    ax = plt.gca()
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                    xx, yy = np.meshgrid(
                        np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50)
                    )

                    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
                    Z = Z.reshape(xx.shape)
                    plt.contour(
                        xx,
                        yy,
                        Z,
                        colors="k",
                        levels=[-1, 0, 1],
                        alpha=0.5,
                        linestyles=["--", "-", "--"],
                    )
                    #########################################################
#                     clf = SVC(kernel='rbf', C=1, random_state=42, max_iter = 1e5).fit(tsne_output, labels)

#                     # Plotting RBF SVM
#                     ax = plt.gca()
#                     xlim = ax.get_xlim()
#                     ylim = ax.get_ylim()
#                     xx, yy = np.meshgrid(
#                         np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50)
#                     )

#                     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#                     Z = Z.reshape(xx.shape)
#                     plt.contour(xx, yy, Z, 
#                                 colors=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'purple', 'black', 'gray', 'lightcoral'], 
#                                 levels=[j + 0.5 for j in range(0,10)], alpha=0.5)
##############################################################################
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
        
        # Creates a cluster for each layer returned by getLayerwiseOutputs
        self.feature_clusters = [SVC(kernel='rbf', C=self.SVC_C, random_state=42, max_iter = 1e5, decision_function_shape='ovr').fit(
                                     x, y) for x in layerwise_outputs]
        
    def predict(self, X):
        layerwise_outputs = self.getLayerwiseOutputs(X)
        
        # Predicts each sample at each layer returned by getLayerwiseOutputs, using the SVM clusters
        return [self.feature_clusters[i].predict(layerwise_outputs[i]) for i in range(len(self.feature_clusters))]
    
    # Modify to put model and tensors on GPU if it's available, then take model off GPU at the end of the method
    def getLayerwiseOutputs(self, X):
        """ Takes a dataset X and runs it through the provided model, saving the output at any linear
            or convolutional layer as it goes.
        """
        X_tensor = torch.utils.data.TensorDataset(torch.Tensor(X))
        X_loader = torch.utils.data.DataLoader(X_tensor, batch_size=128, shuffle=False)
        
        # Maintains a list of each layer's output on each batch of data
        # Small batches of data outputs at each layer are added at a time rather than the output of all of 
        #   the data in a single layer at a time
        layerwise_outputs = []
        
        for x in X_loader:
            x = x[0] # X_loader is expected to return data and labels, but we only want the data
            batch_outputs = self.forward(x)
            # If this is the first batch, just assign it to layerwise outputs
            if len(layerwise_outputs) == 0:
                layerwise_outputs = batch_outputs
            # Otherwise, add each layer in the returned batch to the appropriate layer in layerwise outputs
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