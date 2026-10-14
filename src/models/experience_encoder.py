
import torch.nn as nn


class ExperienceEncoder(nn.Module):

    def __init__(self,input_dim,hidden):
        super().__init__()

        self.net=nn.Sequential(
            nn.Linear(input_dim,hidden),
            nn.ReLU(),
            nn.Linear(hidden,hidden)
        )

    def forward(self,x):
        return self.net(x)
