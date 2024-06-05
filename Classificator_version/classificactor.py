import torch.nn as nn
import torch
import torchvision.transforms as tt

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = tt.ToTensor()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, dilation=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, dilation=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, dilation=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, dilation=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, dilation=1),
            nn.BatchNorm2d(256)
        )
        
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(256*11*6, 1), nn.Sigmoid())
    
    def forward(self, x):
        x = self.transform(x)
        x = x.unsqueeze(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.fc(x)
    
def load_classifiactor(path):
    model = Model()
    model.load_state_dict(torch.load(path))
    return model.eval()