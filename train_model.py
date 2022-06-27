import numpy as np
import torch
from torch import nn, optim
from time import time
import argparse

from model import My_VGG as Model
import utils

def train_model(model, optimizer, x_train, x_test, y_train, y_test, epochs=15):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
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

def main(_):
    # Load data
    if args.dataset == 'MNIST':
        X_train, y_train, X_test, y_test = utils.load_mnist()
        channels, size, classes = 1, 28, 10
    elif args.dataset == 'FMNIST':
        X_train, y_train, X_test, y_test = utils.load_fashion_mnist()
        channels, size, classes = 1, 28, 10
    elif args.dataset == 'GRAY_CIFAR10':
        X_train, y_train, X_test, y_test = utils.load_cifar10()
        X_train, X_test = utils.RGB_to_gray(X_train), utils.RGB_to_gray(X_test)
        channels, size, classes = 1, 32, 10
    else:
        print("Invalid dataset. Valid datasets are 'MNIST', 'FMNIST', and 'GRAY_CIFAR10'")
        return
    
    # Create model and optimizer
    model = Model(in_channels=channels, in_size=size, num_classes=classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train base model
    model, optimizer = train_model(model, optimizer, X_train, X_test, y_train, y_test, epochs=args.initial_epochs)
    
    # Save trained model
    PATH = 'trained_models/' + args.dataset + '_model.pt'
    
    torch.save({'epoch': args.initial_epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'in_channels': channels,
                'in_size': size,
                'num_classes': classes
                }, PATH)
    
    # Generate Adversarial Samples (for adversarial training)
    adversarial_training_eps = 1.2e-1
    
    train_dataloader = utils.create_dataloader(X_train, y_train)
    test_dataloader = utils.create_dataloader(X_test, y_test)
    adv_train_data = utils.add_adversarial_noise(model, train_dataloader, eps=adversarial_training_eps)
    adv_test_data = utils.add_adversarial_noise(model, test_dataloader, eps=adversarial_training_eps)
    
    # Continue training model with adversarial samples
    model, optimizer = train_model(model, optimizer, adv_train_data, adv_test_data, y_train, y_test, epochs=args.adversarial_epochs)
    
    # Save adversarially trained model (separately from initial model)
    PATH = 'trained_models/adv_trained_' + args.dataset + '_model.pt'
    
    torch.save({'epoch': args.initial_epochs + args.adversarial_epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'in_channels': channels,
                'in_size': size,
                'num_classes': classes
                }, PATH)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dataset", type=str, default="MNIST",
                        help="Dataset to train model on. Options are: 'MNIST', 'FMNIST', and 'GRAY_CIFAR10'")
    parser.add_argument("--initial_epochs", type=int, default=15,
                        help="Number of epochs to train initial model for.")
    parser.add_argument("--adversarial_epochs", type=int, default=5,
                        help="Number of epochs to continue training initial model with adversarial samples for.")
    
    args = parser.parse_args()
    main(args)