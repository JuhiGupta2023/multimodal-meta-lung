# src/utils/models.py
"""
Model utilities: 2D encoder wrapper, 3D backbone, fusion head.
Used both by alignment and protonet scripts.
"""
import torch, torch.nn as nn, torch.nn.functional as F

def make_resnet18_backbone(weights=None):
    from torchvision.models import resnet18
    net = resnet18(weights=weights)
    net.fc = nn.Identity()
    return net

class ResNet18_2D_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        net = make_resnet18_backbone(weights=None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = net.layer1, net.layer2, net.layer3, net.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1,3,1,1)
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return self.pool(x).flatten(1)  # [B,512]

class BasicBlock3D(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(c_out)
        self.conv2 = nn.Conv3d(c_out, c_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(c_out)
        self.down = None
        if c_in != c_out or stride != 1:
            self.down = nn.Sequential(nn.Conv3d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm3d(c_out))
    def forward(self, x):
        idn = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.down is not None: idn = self.down(idn)
        return F.relu(x + idn)

class Backbone3D(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(1,32,7,stride=2,padding=3,bias=False), nn.BatchNorm3d(32), nn.ReLU(), nn.MaxPool3d(3,stride=2,padding=1))
        self.l1 = BasicBlock3D(32,64, stride=2)
        self.l2 = BasicBlock3D(64,128, stride=2)
        self.l3 = BasicBlock3D(128,256, stride=2)
        self.pool = nn.AdaptiveAvgPool3d(1)
    def forward(self, x):
        x = self.stem(x); x = self.l1(x); x = self.l2(x); x = self.l3(x)
        return self.pool(x).flatten(1)  # [B,256]

class FusionProtoHead(nn.Module):
    def __init__(self, in_dim=768, hid=512, out_dim=256):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(inplace=True), nn.Linear(hid, out_dim))
        self.scale = nn.Parameter(torch.tensor(10.0))
    def forward(self, e):
        return F.normalize(self.proj(e), dim=1)
    def logits(self, q, p):
        d = torch.cdist(q, p, p=2.0)**2
        return - self.scale * d
