from torch import nn

class FCNet(nn.Module):
    def __init__(self, input_size):
        super(FCNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)