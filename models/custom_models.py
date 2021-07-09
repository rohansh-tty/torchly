import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomResNet(nn.Module):
  def __init__(self, config):
    super(CustomResNet, self).__init__()
    self.config = config
    self.prep_layer = nn.Sequential(
                                nn.Conv2d(3,64, 3,stride=1,padding=1), 
                                nn.BatchNorm2d(64),
                                nn.ReLU(),
                                nn.Dropout(self.config.dropout_value)
        )
    self.layer1x = self.xconv(64, 128)
    self.layer1res = self.resblock(64, 128, self.layer1x)

    self.layer2x = self.xconv(128, 256)

    self.layer3x = self.xconv(256, 512)
    self.layer3res = self.resblock(256, 512, self.layer3x)

    self.gap = nn.AdaptiveAvgPool2d(1)
    self.last_but_one_conv = nn.Conv2d(512, 256, 1)
    self.last_conv = nn.Conv2d(256, 10,1)
  
  def xconv(self, in_ch, out_ch):
    self._layer = nn.Sequential(
                                nn.Conv2d(in_ch,out_ch, 3,stride=1,padding=1), 
                                nn.MaxPool2d(4,stride=2),
                                nn.BatchNorm2d(out_ch),
                                nn.ReLU(),
                                nn.Dropout(self.config.dropout_value)
        )
    return self._layer
    
  def resblock(self, in_ch, out_ch,  x_layer):
    self.block = nn.Sequential(
                                nn.Conv2d(in_ch,out_ch, 3, stride=1), 
                                nn.BatchNorm2d(out_ch),
                                nn.ReLU(),
                                nn.Dropout(self.config.dropout_value),
                                nn.Conv2d(out_ch,out_ch, 2, stride=2), 
                                nn.BatchNorm2d(out_ch),
                                nn.ReLU(),
                                nn.Dropout(self.config.dropout_value)
        )
    self_resblock = self.block
    return self_resblock


  def forward(self, input):
    prep_op = self.prep_layer(input) 

    layer1x = self.layer1x(prep_op)
    layer1res = self.layer1res(prep_op)
    resblock_layer1 = layer1x + layer1res
    layer1_op = layer1x+layer1res

    layer2op = self.layer2x(layer1_op)

    layer3x = self.layer3x(layer2op)
    layer3res = self.layer3res(layer2op)
    resblock_layer3 = layer3x + layer3res
    layer3_op = layer3x+layer3res
    
    gap = self.gap(layer3_op)

    last_but_one = self.last_but_one_conv(gap)
    last_one = self.last_conv(last_but_one)
   
    final_op = last_one.view(last_one.shape[0],-1)

    if self.config.loss_function == 'CrossEntropyLoss':
      return final_op
    elif self.config.loss_function == 'NLLLoss':
      return F.log_softmax(final_op, dim=-1)

