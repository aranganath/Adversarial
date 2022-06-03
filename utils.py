import torchvision
from torchvision import transforms
import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from time import time

from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10

from model import My_VGG

""" Each 'load' function loads the indicated dataset, converts it to float, and 
    converts the data range to [0, 1]. If necessary, the data is also reshaped
    to a shape that can be used by a PyTorch model.
"""
def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, axis=1)
    x_test = np.expand_dims(x_test, axis=1)
    
    return normalize_data(x_train, y_train, x_test, y_test)
    
def load_fashion_mnist():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = np.expand_dims(x_train, axis=1)
    x_test = np.expand_dims(x_test, axis=1)
    
    return normalize_data(x_train, y_train, x_test, y_test)

def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.reshape((-1,3,32,32))
    x_test = x_test.reshape((-1,3,32,32))
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)
    
    return normalize_data(x_train, y_train, x_test, y_test)

def normalize_data(X_train, y_train, X_test, y_test):
    # Conversion to float
    X_train = X_train.astype('float32') 
    X_test = X_test.astype('float32')
    # Normalization
    X_train = X_train/255.0
    X_test = X_test/255.0
    # Flatten
    #X_train = x_train.reshape(len(x_train),-1)
    #X_test = x_test.reshape(len(x_test),-1)
    
    return X_train, y_train, X_test, y_test

def train_model(model, x_train, x_test, y_train, y_test, epochs=15):
    """ Trains a model on the training data provided, and then evaluates it on the test data.
        Also calls view_classify on a sample image from the test set, which displays the image 
        itself and the model's output for that image.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #transform = transforms.Compose([transforms.Normalize((0,), (1,)),])
    
    trainset = torch.utils.data.TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train).type(torch.long))
    valset = torch.utils.data.TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test).type(torch.long))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    if device == "cuda":
        model = model.cuda()
    model.train()
    
    time0 = time()
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
    print("\nTraining Time (in minutes) =",(time()-time0)/60)

    model.to("cpu")
    images, labels = next(iter(valloader))

    img = images[0].unsqueeze(axis=0)
    with torch.no_grad():
        logps = model(img)

    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    print("Predicted Class =", probab.index(max(probab)))
    view_classify(img.squeeze(), ps)
    
    model.eval()
    if device == "cuda":
        model = model.cuda()
    correct_count, all_count = 0, 0
    for x, y in valloader:
        all_count += y.shape[0]
        x, y = x.to(device), y.to(device)
        output = model(x)
        preds = output.max(1)[1]
        correct_count += preds.eq(y).sum().item()
    model.to("cpu")

    print("Number Of Images Tested =", all_count)
    print("\nModel Accuracy =", (correct_count/all_count))
    
    return model

def load_VGG(path, load_checkpoint=False):
    """ Used to load a pre-trained model. Right now it's set up to specifically load an instance of
        the My_VGG model. Setting the load_checkpoint parameter to True will make the function return
        the model, the state of the optimizer at the point the model stopped training, and the epoch
        it stopped training at. Leaving load_checkpoint as False will only return the loaded model.
    """
    checkpoint = torch.load(path)
    
    channels, size, classes = checkpoint['in_channels'], checkpoint['in_size'], checkpoint['num_classes']
    model = My_VGG(in_channels=channels, in_size=size, num_classes=classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if load_checkpoint:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        return model, optimizer, epoch
    
    return model

def eval_model(model, data, labels, high_conf_thres=None):
    """ Given a PyTorch model, ndarray of data points, and labels, prints model accuracy on the data.
        If a high_conf_thres is provided, it will be used as a threshold for what is considered a 
        "high confidence prediction" by the model. This will be used to report the proportion of 
        incorrectly classified samples for which the model had "high confidence" (class probability 
        greater than or equal to high_conf_thres)
    """
    correct_count, all_count, high_conf_misclassification_count = 0, 0, 0
    
    if high_conf_thres != None:
        for i in range(len(data)):
            img = torch.from_numpy(data[i]).unsqueeze(axis=0) # Create tensor with batch dimension
            with torch.no_grad():
                logps = model(img)

            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            pred_label = probab.index(max(probab))
            if (pred_label == labels[i]):
                correct_count += 1
            elif ((high_conf_thres != None) and (ps.max().item() >= high_conf_thres)):
                high_conf_misclassification_count += 1
            all_count += 1

        print("Number Of Samples Tested =", all_count)
        print("Model Accuracy =", (correct_count/all_count))
        print("\nNumber of misclassified samples with high model confidence = ", (high_conf_misclassification_count))
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        testset = torch.utils.data.TensorDataset(torch.Tensor(data), torch.Tensor(labels).type(torch.long))
        testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
        
        model.eval()
        if device == "cuda":
            model = model.cuda()
            
        for x, y in testloader:
            all_count += y.shape[0]
            x, y = x.to(device), y.to(device)
            logps = model(x)
            output = model(x)
            preds = output.max(1)[1]
            correct_count += preds.eq(y).sum().item()
        model.to("cpu")
        return (correct_count/all_count)

def get_network_outputs(model, dataloader):
    """ Given a model and dataloader, returns an ndarray of the model outputs for data in the dataloader.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model = model.cuda()
    model.eval()
    network_outputs = list()
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            network_outputs.append(model(x).cpu())
    model.to("cpu")
            
    network_outputs = np.concatenate(network_outputs)
    
    return network_outputs

def add_adversarial_noise(model, dataloader, eps=3e-2):
    """ Returns an ndarray of adversarially perturbed versions of the data passed in with the dataloader, 
        generated by a white-box FGSM attack on the provided network.
    """
    adv_images = list()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        adv_images.append(fast_gradient_method(model, x, eps, np.inf).detach().cpu().numpy())
            
    adv_images = np.concatenate(adv_images)
    if device == "cuda":
        model.to("cpu")
    
    return adv_images

def create_dataloader(data, labels=None):
    """ Returns PyTorch dataloader object created from passed in data
    """
    if (labels is not None):
        dataset = torch.utils.data.TensorDataset(torch.Tensor(data), torch.Tensor(labels).type(torch.long))
    else:
        dataset = torch.utils.data.TensorDataset(torch.Tensor(data))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    
    return dataloader

def one_vs_all_dataloader(data, labels, digit):
    """ Returns dataloader with labels modified so that samples belonging to the specified class have a 
        label of 1, and all other samples have a label of 0. Also returns an ndarray of modified labels
    """
    one_v_all_labels = np.zeros(labels.shape, dtype=labels.dtype)
    
    current_digit_idx = np.where(labels == digit)
    one_v_all_labels[current_digit_idx] = 1
    
    dataset = torch.utils.data.TensorDataset(torch.Tensor(data), torch.Tensor(labels).type(torch.long))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    
    return dataloader, one_v_all_labels

def view_classify(img, ps, img_title=None):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    if (img.shape[0] == 3):
        ax1.imshow(img.reshape(32,32,3))
    else:
        ax1.imshow(img.reshape(28,28))
    ax1.axis('off')
    if img_title is not None:
        ax1.set_title(img_title)
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()