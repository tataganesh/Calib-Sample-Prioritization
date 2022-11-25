import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch
import resnet
import wideresnet


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.fc1 = nn.Linear(3136, 8)
        self.fc2 = nn.Linear(8, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        output = self.fc2(x)
        return output


def get_resnet_model(model_name, num_classes):
    model = model_mapping[model_name](num_classes)
    return model


def get_model(model_name, num_classes=10):
    if 'wideresnet' in model_name:
        depth, widen_factor = [int(x) for x in model_name.split('-')[1:]]
        return wideresnet.WideResNet(depth=depth, widen_factor=widen_factor, num_classes=10)
    elif 'resnet' in model_name:
        return get_resnet_model(model_name, num_classes=num_classes)


model_mapping = {
    'vgg_16': torchvision.models.vgg16,
    'vgg_11': torchvision.models.vgg11,
    "resnet_18": resnet.ResNet18,
    "resnet_34": resnet.ResNet34,
    "resnet_50": resnet.ResNet50
}
