import numpy as np
import torch
from torch import nn, optim
from time import time

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

def main():
    # Load data
    X_train, y_train, X_test, y_test = utils.load_mnist()
    channels, size, classes = 1, 28, 10
    
    # Create model and optimizer
    model = Model(in_channels=channels, in_size=size, num_classes=classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train base model
    epochs = 15
    model, optimizer = train_model(model, optimizer, X_train, X_test, y_train, y_test, epochs=epochs)
    
    # Save trained model
    PATH = 'trained_models/initial_model.pt'
    
    torch.save({'epoch': epochs,
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
    model, optimizer = train_model(model, optimizer, adv_train_data, adv_test_data, y_train, y_test, epochs=5)
    
    # Save adversarially trained model (separately from initial model)
    PATH = 'trained_models/baseline_model.pt'
    
    torch.save({'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'in_channels': channels,
                'in_size': size,
                'num_classes': classes
                }, PATH)
    
if __name__ == "__main__":
    
    main()