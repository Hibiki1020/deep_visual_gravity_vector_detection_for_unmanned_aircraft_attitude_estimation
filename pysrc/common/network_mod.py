import torch
from torchvision import models
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, resize, use_pretrained_vgg=True):
        super(Network, self).__init__()

        self.kernel_size = 3

        vgg = models.vgg16(pretrained=use_pretrained_vgg)
        #vgg = models.VGG()
        self.cnn_feature = vgg.features

        self.cnn = nn.Sequential (
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.cnn_add = nn.Sequential(
            nn.Conv2d(   512,  64, self.kernel_size, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        dim_fc_in = 12 * 12 * 64
        self.cnn = nn.Sequential(
            self.block1_output,
            self.block2_output,
            self.block3_output,
            self.block4_output,
            self.block5_output
        )
        self.fc = nn.Sequential(
            nn.Linear(dim_fc_in, 18),
            nn.ReLU(inplace=True),
            nn.Linear(18, 3)
        )

    def initializeWeights(self):
        for m in self.fc.children():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def getParamValueList(self):
        list_cnn_param_value = []
        list_cnn_add_param_value = []
        list_fc_param_value = []
        
        for param_name, param_value in self.named_parameters():
            param_value.requires_grad = True
            if "cnn" in param_name:
                # print("cnn: ", param_name)
                list_cnn_param_value.append(param_value)
            if "cnn_add" in param_name:
                list_cnn_add_param_value.append(param_value)
            if "fc" in param_name:
                # print("fc: ", param_name)
                list_fc_param_value.append(param_value)
        # print("list_cnn_param_value: ",list_cnn_param_value)
        # print("list_fc_param_value: ",list_fc_param_value)
        return list_cnn_param_value, list_cnn_add_param_value, list_fc_param_value

    def forward(self, x):
        #x = self.cnn_feature(x)
        x = self.cnn(x)
        x = self.cnn_add(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        l2norm = torch.norm(x[:, :3].clone(), p=2, dim=1, keepdim=True)
        x[:, :3] = torch.div(x[:, :3].clone(), l2norm)  #L2Norm, |(gx, gy, gz)| = 1
        return x
