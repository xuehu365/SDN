from torchvision import models
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torchvision.models.resnet import Bottleneck
from models.lowlevel_randomizations import *


__all__ = ['resnet50', 'resnet101']
url = ["https://download.pytorch.org/models/resnet50-11ad3fa6.pth","https://download.pytorch.org/models/resnet101-63fe2227.pth"]

class ResNet(models.ResNet):

    def __init__(self, num_classes, feat_dim, *args, **kwargs):
        super(ResNet, self).__init__(*args, **kwargs)
        self._out_features = self.fc.in_features
        del self.fc

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Sequential(
            nn.Linear(self._out_features, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(feat_dim, num_classes, bias=False)
        self.stylemix = StyleInjectTtoS()

        self.flatten = nn.Flatten()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        if self.training:
            x_ex = self.stylemix(x)
        else:
            x_ex = x
        x_ex = self.layer3(x_ex)
        x_ex = self.layer4(x_ex)
        x_ex = self.avgpool(x_ex)
        x_ex = self.flatten(x_ex)
        f = self.fc1(x_ex)
        pred = self.fc2(f)
        
        return f, pred

    @property
    def out_features(self) -> int:
        return self._out_features

def _resnet(num_classes, feat_dim, arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(num_classes, feat_dim, block, layers, **kwargs)

    if arch=='resnet50':
        model_url = url[0]
    else:
        model_url = url[1]

    if pretrained:
        state_dict = load_state_dict_from_url(model_url, progress=progress)
        model.load_state_dict(state_dict, strict=False)
    
    return model

def resnet50(args, pretrained=True, progress=True, **kwargs):
    return _resnet(args.num_classes, args.feat_dim, 'resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

def resnet101(args, pretrained=True, progress=True, **kwargs):
    return _resnet(args.num_classes, args.feat_dim, 'resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)

def resnet50_cam(pretrained=True, progress=True, **kwargs):
    return _resnet(65, 1024, 'resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)
