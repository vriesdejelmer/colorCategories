"""
Wrapper for classification.
"""

from torch import nn, flatten


class ClassModel(nn.Module):


    def __init__(self, backbone, output_num):
        super(ClassModel, self).__init__()

        self.backbone = backbone
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512, out_features=output_num)

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = flatten(x, 1)
        x = self.fc(x)

#        x = self.fc(x.view(-1))
        return x

