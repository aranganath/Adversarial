import torch
import torch.nn as nn

class My_VGG(nn.Module):
    def __init__(self, in_channels=3, in_size=28, num_classes=10):
        super().__init__()
        features_cfg = [64, "M", 128, "M", 256, 256, "M", 512, 512]
        classifier_cfg = [4096, "ReLU", "Dropout", 4096, "ReLU", "Dropout", num_classes]
        input_channels = in_channels
        input_size = in_size
        
        layers = []
        for v in features_cfg:
            if isinstance(v, int):
                layers.append(nn.Conv2d(input_channels, v, kernel_size=3, stride=1, padding=1, bias=True))
                input_channels = v
                layers.append(nn.BatchNorm2d(input_channels))
                layers.append(nn.ReLU(inplace=True))
            elif v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False))
                input_size = int(input_size / 2)
        self.features = nn.Sequential(*layers)
        
        layers = []
        layers.append(nn.Flatten(start_dim=1, end_dim=-1))
        for v in classifier_cfg:
            if isinstance(v, int):
                layers.append(nn.Linear(input_channels * input_size**2, v, bias=True))
                input_size = 1
                input_channels = v
            elif v == "ReLU":
                layers.append(nn.ReLU(inplace=True))
            elif v == "Dropout":
                layers.append(nn.Dropout())
        layers.append(nn.LogSoftmax(dim=1))
        self.classifier = nn.Sequential(*layers)
        
        self.apply(self.init_weights)
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def init_weights(self, m):
        with torch.no_grad():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)