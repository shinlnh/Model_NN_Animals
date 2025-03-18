import torch
import torch.nn as nn


class SimpleNeuralNetwork(nn.Module):
    def __init__(self, num_class = 10):

        super().__init__()

        self.flatten = nn.Flatten()

        self.m1 = nn.Sequential(
            nn.Linear(in_features=3*32*32,out_features=256),

            nn.ReLU()
        )
        self.m2 = nn.Sequential(
            nn.Linear(in_features=256, out_features=512),

            nn.ReLU()
        )
        self.m2 = nn.Sequential(
            nn.Linear(in_features=256, out_features=512),

            nn.ReLU()
        )
        self.m3 = nn.Sequential(
            nn.Linear(in_features=512, out_features=1024),

            nn.ReLU()
        )
        self.m4 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),

            nn.ReLU()
        )
        self.m5 = nn.Sequential(
            nn.Linear(in_features=512, out_features=num_class),

            nn.ReLU()
        )

    def forward(self,x):
        x = self.flatten(x)
        x = self.m1(x)
        x = self.m2(x)
        x = self.m3(x)
        x = self.m4(x)
        x = self.m5(x)
        return x

if __name__ == '__main__':

    model = SimpleNeuralNetwork()
    input_data = torch.rand(8,3,32,32)
    result = model(input_data)
    print(result)
