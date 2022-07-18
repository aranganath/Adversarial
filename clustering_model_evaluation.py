import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import pickle
from time import time

# Input space transformation (dimensionality reduction)
from DCTDimReducer import DCTDimReducer

import utils
from model import My_VGG as Model

from ClusteringAdvClassifier import ClusterAdversarialClassifier as Classifier

def main():
    for i in range(10):
        results = dict()

        datasets = ['MNIST', 'FMNIST']
        for dataset in datasets:
            for adversarial in [False, True]:
                entry_name = 'adversarial_' if adversarial else 'clean_'
                entry_name = entry_name + dataset
                print(entry_name)

                (runtime, baseline_performance, clustering_classifier_performance, clustering_detector_performance, 
                 cluster_accuracy, cnn_accuracy, baseline_prediction_times, cluster_prediction_times) = run(dataset, adversarial)
                entry = {'runtime': runtime, 'baseline_performance': baseline_performance, 
                         'clustering_classifier_performance': clustering_classifier_performance, 
                         'clustering_detector_performance': clustering_detector_performance, 
                         'cluster_accuracy': cluster_accuracy, 'cnn_accuracy': cnn_accuracy, 
                         'baseline_prediction_times': baseline_prediction_times, 'cluster_prediction_times': cluster_prediction_times}
                results[entry_name] = entry

        print(results)

        fileObj = open('results/results' + str(i) + '.obj', 'wb')
        pickle.dump(results,fileObj)
        fileObj.close()

# Valid datasets are 'MNIST', 'FMNIST', and 'GRAY_CIFAR10'
# Adversarial is true or false, indicating whether the base model was adversarially trained or not
def run(dataset, adversarial, dct_size_reduction=16, SVC_C=100):
    
    # Load data
    if dataset == 'MNIST':
        X_train, y_train, X_test, y_test = utils.load_mnist(normalize=True)
        channels, size, classes = 1, 28, 10
    elif dataset == 'FMNIST':
        X_train, y_train, X_test, y_test = utils.load_fashion_mnist(normalize=True)
        channels, size, classes = 1, 28, 10
    elif dataset == 'GRAY_CIFAR10':
        X_train, y_train, X_test, y_test = utils.load_cifar10(normalize=True, grayscale=True)
        channels, size, classes = 1, 32, 10
    dimensions = (size, size)

    if adversarial:
        model_name = 'adv_trained_'+ dataset + '_model'
    else:
        model_name = dataset + '_model'
    
    # Load baseline (standalone model)
    baseline_model = utils.load_VGG('trained_models/' + model_name + '.pt')
    baseline_model.eval()
    # Load model to be wrapped with clustering detector/classifier
    model = utils.load_VGG('trained_models/' + model_name + '.pt')
    model.eval()
    
    # Discrete Cosine Transform for dimensionality reduction
    transformer = DCTDimReducer(coef_size_reduction=dct_size_reduction)
    
    # Fitting the clustering classifier on clean data
    classifier = Classifier(model, input_transform=transformer, SVC_C = SVC_C)
    classifier.fit(X_train, y_train)
    
    # Tracking training times
    runtime = dict()
    runtime['input_cluster_train_time'] = classifier.input_cluster_train_time
    runtime['output_cluster_train_time'] = classifier.output_cluster_train_time
    runtime['dim_reduction_time'] = transformer.transform_time
    
    # Evaluating models
    test_eps = ['clean', .1, .3, .6, 1, 1.5, 2, 3]

    baseline_performance = dict()
    clustering_classifier_performance = dict()
    cluster_accuracy = dict()
    cnn_accuracy = dict()
    clustering_detector_performance = dict()
    baseline_prediction_times = dict()
    cluster_prediction_times = dict()

    test_dataloader = utils.create_dataloader(X_test, y_test)
    
    runs = len(test_eps)
    for eps in test_eps:
        # Track total prediction time to get average prediction time for each epsilon
        total_cluster_time = 0
        total_baseline_time = 0
        
        if eps == 'clean':
            adv_test_data = X_test
            print("\nCurrent epsilon: No adversarial noise")
        else:
            adv_test_data = utils.add_adversarial_noise(model, test_dataloader, eps=eps)
            print("\nCurrent epsilon: %.2f" % (eps))

        # Testing both classifiers on adversarial data only
        # Baseline accuracy
        time0 = time()
        baseline_performance[str(eps)] = utils.eval_model(baseline_model, adv_test_data, y_test)
        total_baseline_time = total_baseline_time + time() - time0

        print("\tBaseline classifier accuracy: %.3f" % (baseline_performance[str(eps)]))

        # Clustering classifier and adversarial detection accuracy
        time0 = time()
        clustering_classifier_performance[str(eps)] = classifier.score(adv_test_data, y_test)
        total_cluster_time = total_cluster_time + time() - time0
        
        cluster_accuracy[str(eps)] = classifier.cluster_accuracy
        cnn_accuracy[str(eps)] = classifier.cnn_accuracy
        if eps == 'clean':
            clustering_detector_performance[str(eps)] = 1 - classifier.proportion_flagged
        else:
            clustering_detector_performance[str(eps)] = classifier.proportion_flagged

        print("\n\tCluster classifier accuracy: %.3f" % (clustering_classifier_performance[str(eps)]))
        print("\tClustering algorithm accuracy: %.3f" % (cluster_accuracy[str(eps)]))
        print("\tModel accuracy: %.3f" % (cnn_accuracy[str(eps)]))
        print("\tInput data flagged as suspicious by detector: %.3f" % (clustering_detector_performance[str(eps)]))
        
        baseline_prediction_times[str(eps)] = total_baseline_time / runs
        cluster_prediction_times[str(eps)] = total_cluster_time / runs
    
    return runtime, baseline_performance, clustering_classifier_performance, clustering_detector_performance, cluster_accuracy, cnn_accuracy, baseline_prediction_times, cluster_prediction_times

if __name__ == "__main__":
    
    main()