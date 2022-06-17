import numpy as np
from sklearn.svm import SVC
from scipy import fftpack

import utils

class ClusterAdversarialClassifier():
    def __init__(self, model, input_transform=None, SVC_C=1):
        self.model = model
        self.input_transform = input_transform
        self.SVC_C = SVC_C
        
    def fit(self, X, y):
        # Find RBF classification boundary in input space
        X_flat = X.reshape(X.shape[0], -1)
        if self.input_transform:
            self.input_cluster = SVC(kernel='rbf', C=self.SVC_C, random_state=42, max_iter = 1e5, decision_function_shape='ovr').fit(
                                     self.input_transform.fit_transform(X_flat), y)
        else:
            self.input_cluster = SVC(kernel='rbf', C=self.SVC_C, random_state=42, max_iter = 1e5, decision_function_shape='ovr').fit(
                                     X_flat, y)

        # Get model outputs
        train_dataloader = utils.create_dataloader(X, y)
        outputs = utils.get_network_outputs(self.model, train_dataloader)
        
        # Find RBF classification boundary in output space
        self.output_cluster = SVC(kernel='rbf', C=self.SVC_C, random_state=42, max_iter = 1e5, decision_function_shape='ovr').fit(
                                  outputs, y)
        
    def predict(self, X):
        # Get model outputs on test data
        test_dataloader = utils.create_dataloader(X, np.zeros(X.shape[0], dtype=np.uint8))
        outputs = utils.get_network_outputs(self.model, test_dataloader)
        # Get model preds from outputs
        preds = np.argmax(np.exp(outputs), axis=1)
        
        # Get input and output cluster predictions - if these disagree, we will consider the sample suspicious
        X_flat = X.reshape(X.shape[0], -1)
        if self.input_transform:
            try:
                input_cluster_preds = (self.input_cluster).predict(self.input_transform.transform(X_flat))
            except AttributeError:
                input_cluster_preds = (self.input_cluster).predict(self.input_transform.fit_transform(X_flat))
        else:
            input_cluster_preds = (self.input_cluster).predict(X_flat)
        output_cluster_preds = (self.output_cluster).predict(outputs)
        flagged_sample_indices = (input_cluster_preds != output_cluster_preds)
        
        self.proportion_flagged = (input_cluster_preds != output_cluster_preds).sum().item() / X.shape[0]
        
        # For samples flagged as likely to be adversarial by clusters, predict using instead the input cluster classification
        preds[flagged_sample_indices] = input_cluster_preds[flagged_sample_indices]
        
        self.flagged_sample_indices = flagged_sample_indices
        
        return preds

    def score(self, X, y):
        # Get predictions
        preds = self.predict(X)
        
        cluster_sample_indices = self.flagged_sample_indices
        cnn_sample_indices = np.logical_not(self.flagged_sample_indices)
        
        self.cluster_accuracy = (preds[cluster_sample_indices] == y[cluster_sample_indices]).sum().item() / len(y[cluster_sample_indices])
        self.cnn_accuracy = (preds[cnn_sample_indices] == y[cnn_sample_indices]).sum().item() / len(y[cnn_sample_indices])
        
        # Return proportion that are correct
        return (preds == y).sum().item() / y.shape[0]