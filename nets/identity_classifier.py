from .modules import BasicModule
from torch import nn


class IdentityClassifier(BasicModule):

    def __init__(self, num_id):
        super(IdentityClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Dropout(.7),  # source code using keep_prob=0.3
            nn.Linear(256, num_id))

    def forward(self, x):
        return self.model(x)
