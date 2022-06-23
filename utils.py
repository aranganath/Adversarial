import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from matplotlib.text import OffsetFrom
from matplotlib.patches import FancyArrowPatch

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from time import time

#from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10

from model import My_VGG

""" Each 'load' function loads the indicated dataset, converts it to float, and 
    converts the data range to [0, 1]. If necessary, the data is also reshaped
    to a shape that can be used by a PyTorch model.
"""
def load_mnist():
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = torchvision.datasets.MNIST("data", download=True, train=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST("data", download=True, train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
    
    (train_data, train_labels), (test_data, test_labels) = next(iter(train_loader)), next(iter(test_loader))
    x_train, y_train = train_data.numpy(), train_labels.numpy()
    x_test, y_test = test_data.numpy(), test_labels.numpy()
    
    return x_train, y_train, x_test, y_test
    
def load_fashion_mnist():
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = torchvision.datasets.FasionMNIST("data", download=True, train=True, transform=transform)
    test_dataset = torchvision.datasets.FasionMNIST("data", download=True, train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
    
    (train_data, train_labels), (test_data, test_labels) = next(iter(train_loader)), next(iter(test_loader))
    x_train, y_train = train_data.numpy(), train_labels.numpy()
    x_test, y_test = test_data.numpy(), test_labels.numpy()
    
    return x_train, y_train, x_test, y_test

def load_cifar10():
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = torchvision.datasets.CIFAR10("data", download=True, train=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10("data", download=True, train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
    
    (train_data, train_labels), (test_data, test_labels) = next(iter(train_loader)), next(iter(test_loader))
    x_train, y_train = train_data.numpy(), train_labels.numpy()
    x_test, y_test = test_data.numpy(), test_labels.numpy()
    
    return x_train, y_train, x_test, y_test

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
        ax1.imshow(np.transpose(img, (1,2,0)))
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

def visualize_full_binary_tree(max_depth, box_labels=None, box_values=None, arrow_text=None, ax=None, text_size='small', 
                               offset=0, depth=0, i=1):
    if depth == max_depth: # Base case
        return
    
    # If no axes are provided, create them
    if ax==None:
        _, ax = plt.subplots()
        ax.set_axis_off()
    
    # Constants defining the size of gaps between boxes in the visualization
    vertical_gap = .1
    horizontal_gap = .025
    
    # Define sizes of boxes (representing nodes in tree) relative to remaining space at current depth of tree
    box_height = (.99 - vertical_gap*max_depth) / max_depth
    remaining_height = 1 - (box_height + vertical_gap) * (depth+1)
    
    box_width = 1 / 2**(max_depth-1) - horizontal_gap
    curr_width = 1 / 2**depth
    
    # Draw left subtree
    visualize_full_binary_tree(max_depth, box_labels, box_values, arrow_text, ax, text_size, offset, depth + 1, i*2)
    
    # Draw right subtree
    visualize_full_binary_tree(max_depth, box_labels, box_values, arrow_text, ax, text_size, offset+curr_width/2, depth + 1, i*2+1)
    
    # Draw arrows between node at current level and left and right children
    if depth + 1 != max_depth:
        origin = (curr_width/2 + offset, remaining_height)
        destination = (curr_width/4 + offset, remaining_height - vertical_gap)
        # Used to fill text next to arrows
        left_arrow_text = ''
        right_arrow_text = ''
        
        if arrow_text != None:
            # Reads arrow_text values in top-down, left-right order
            left_arrow_text = str(arrow_text[(i*2) - 2])
            right_arrow_text = str(arrow_text[(i*2 + 1) - 2])
        
        # Draw left arrow
        left_arrow = FancyArrowPatch(origin, destination, mutation_scale=8, arrowstyle="->")
        ax.add_patch(left_arrow)
        ax.annotate(left_arrow_text, # Draw text for left arrow
                    xy=destination,
                    xytext=(0, 0), textcoords=OffsetFrom(left_arrow, (0.5, 0.5), "pixels"),
                    ha="right", fontsize=text_size)
        
        # Draw right arrow
        right_arrow = FancyArrowPatch(origin, (destination[0]+curr_width/2, destination[1]), mutation_scale=8, arrowstyle="->")
        ax.add_patch(right_arrow)
        ax.annotate(right_arrow_text, # Draw text for right arrow
                    xy=destination,
                    xytext=(0, 0), textcoords=OffsetFrom(right_arrow, (0.5, 0.5), "pixels"),
                    ha="left", fontsize=text_size)
        
    # Draw node at current level
    x = (curr_width - box_width) / 2 + offset
    y = remaining_height
    node = plt.Rectangle((x, y), box_width, box_height, fill=False)
    ax.add_patch(node)
    
    # In-box annotation
    if box_labels != None:
        ax.text(curr_width/2 + offset, remaining_height+box_height-.01, str(box_labels[i-1]),
                horizontalalignment='center',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=text_size, wrap=True)
    
    if box_values != None:
        ax.text(x + box_width / 2, y + box_height / 2, str(box_values[i-1]),
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes, size=text_size, wrap=True)
    
    return ax