import numpy as np
import torch
from torch import nn, optim
from time import time
import argparse
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent

from model import My_VGG as Model
import utils

def train_model(model, optimizer, x_train, x_test, y_train, y_test, epochs=15, adv_train=False, eps=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer_to(optimizer, device)
    
    trainset = torch.utils.data.TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train).type(torch.long))
    valset = torch.utils.data.TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test).type(torch.long))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

    criterion = nn.NLLLoss()
    
    if device == "cuda":
        model = model.cuda()
    model.train()
    
    time0 = time()
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            if adv_train:
                images = projected_gradient_descent(model, images, eps, 0.01, 5, np.inf)
                                                    #clip_min=images.min(), clip_max=images.max())

            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
    print("\nTraining Time (in minutes) =",(time()-time0)/60)
    
    model.eval()
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
    
    return model, optimizer

# https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068/2
def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
def main(_):
    # Load data
    if args.dataset == 'MNIST':
        X_train, y_train, X_test, y_test = utils.load_mnist(normalize=True)
        channels, size, classes = 1, 28, 10
    elif args.dataset == 'FMNIST':
        X_train, y_train, X_test, y_test = utils.load_fashion_mnist(normalize=True)
        channels, size, classes = 1, 28, 10
    elif args.dataset == 'GRAY_CIFAR10':
        X_train, y_train, X_test, y_test = utils.load_cifar10(normalize=True, grayscale=True)
        channels, size, classes = 1, 32, 10
    elif args.dataset == 'CIFAR10':
        X_train, y_train, X_test, y_test = utils.load_cifar10(normalize=True)
        channels, size, classes = 3, 32, 10
    else:
        print("Invalid dataset. Valid datasets are 'MNIST', 'FMNIST', 'CIFAR10', and 'GRAY_CIFAR10'")
        return
    
    if args.adversarial:
        model_name = 'adv_trained_' + args.dataset + '_model.pt'
    else:
        model_name = args.dataset + '_model.pt'
    
    if args.resume_training:
        # Load model and optimizer
        PATH = 'trained_models/' + model_name
        model, optimizer, epoch = utils.load_VGG(PATH, True)
        epochs = args.epochs - epoch
    else:
        # Create model and optimizer
        model = Model(in_channels=channels, in_size=size, num_classes=classes)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        epochs = args.epochs
    
    # Train base model
    model, optimizer = train_model(model, optimizer, X_train, X_test, y_train, y_test, 
                                   epochs, args.adversarial, args.eps)
    
    if args.adversarial:
        eps = args.eps
    else:
        eps = 0
    
    # Save trained model
    if args.save_model:
        PATH = 'trained_models/' + model_name

        torch.save({'epoch': args.epochs,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'in_channels': channels,
                    'in_size': size,
                    'num_classes': classes,
                    'epsilon': eps
                    }, PATH)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dataset", type=str, default="MNIST",
                        help="Dataset to train model on. Options are: 'MNIST', 'FMNIST', 'CIFAR10', and 'GRAY_CIFAR10'")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs to train model for.")
    parser.add_argument("--save_model", type=bool, default=True,
                        help="Save the trained models? Saved in a directory 'trained_models'.")
    parser.add_argument("--resume_training", type=bool, default=False,
                        help="Resume training of an existing model with with dataset in the directory 'trained_models'.")
    parser.add_argument("--adversarial", type=bool, default=False,
                        help="Use PGD to do apply adversarial training to the model.")
    parser.add_argument("--eps", type=float, default=.3,
                        help="Magnitude of adversarial noise to apply to adversarial samples for adversarial training.")
    
    args = parser.parse_args()
    main(args)