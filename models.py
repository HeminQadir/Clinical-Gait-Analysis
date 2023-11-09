
import torch
import torch.nn as nn

class GroupNormConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, groups=groups)
        self.activation = nn.GELU()
        self.layer_norm = nn.GroupNorm(num_groups=out_channels, num_channels=out_channels, eps=1e-05, affine=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.layer_norm(x)
        return x
    

class NoLayerNormConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, groups=groups)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x
    

class FeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()

        out_channels  = int((config.hidden_size)*config.in_channels)

        self.conv_layers = nn.ModuleList([
            GroupNormConvLayer(config.in_channels, out_channels, 3, 1, groups=config.in_channels),
            NoLayerNormConvLayer(out_channels, out_channels, 3, 1, groups=config.in_channels),
            NoLayerNormConvLayer(out_channels, out_channels, 3, 2, groups=config.in_channels),
            NoLayerNormConvLayer(out_channels, out_channels, 3, 2, groups=config.in_channels),
            NoLayerNormConvLayer(out_channels, out_channels, 3, 2, groups=config.in_channels),
            NoLayerNormConvLayer(out_channels, out_channels, 2, 2, groups=config.in_channels),
            NoLayerNormConvLayer(out_channels, out_channels, 2, 2, groups=config.in_channels)
        ])

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
            #print(x.shape)
        return x
    
class FeatureProjection(nn.Module):
    def __init__(self, config):
        super().__init__()

        out_channels  = int((config.hidden_size/2)*config.in_channels)

        self.layer_norm = nn.LayerNorm(out_channels, eps=1e-05)
        self.projection = nn.Linear(out_channels, config.hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.projection(x)
        x = self.dropout(x)
        return x
    
    
class Classification_1DCNN(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.in_channels = config.in_channels
        self.num_classes = config.num_classes
        self.hidden_size = config.hidden_size

        self.feature_extractor = FeatureExtractor(config)
        #self.feature_projection = FeatureProjection(config)
        
        self.classifier = nn.Linear(self.hidden_size*self.in_channels, self.num_classes)
        
    def merged_strategy(self, hidden_states, mode="mean"):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")
        return outputs

    def forward(self, inputs, labels=None, **kwargs):
        #print("="*10)
        #print(inputs.shape)
        #torch.Size([5, 660000])
        
        x = inputs #.unsqueeze(1)
        #print(x.shape)
        #torch.Size([5, 1, 660000])
        
        x = self.feature_extractor(x)
        #print(x.shape)
        #torch.Size([5, 512, 2062])
        
        x = x.transpose(1, 2)
        #print(x.shape)
        #torch.Size([5, 2062, 512])
        
        #x = self.feature_projection(x)
        #print(x.shape)
        #torch.Size([5, 2062, 768])
        
        x = self.merged_strategy(x, mode="mean")
        #print(x.shape)
        #torch.Size([5, 768])
        
        logits = self.classifier(x)
        #print(logits.shape)
        #torch.Size([5, 10])

        if labels is not None:
            # if batch size = 9
            #print("I am in the loss")
            #print(labels.view(-1))
            #print(logits.view(-1, self.num_classes).shape)
            #loss_fct = BCEWithLogitsLoss()
            #loss =  loss_fct(logits.squeeze(), labels.float())#
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1)) #loss_fct(logits, labels) #
            #print(loss)
            return loss 
        else:
            return logits

