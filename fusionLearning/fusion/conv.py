from fusion.interfaces import FusionModule

class ConvConfig(): 
    def __init__(self,
                 in_channels = 3,
                 out_channels = 3,
                 kernel_size = 3,
                 padding = 1,
                 stride = 1,
                 dilation = 1,
                 groups = 1,
                 bias = True,
                 padding_mode = 'zeros',
                 subsample = False,
                 use_batch_norm = False,
                 use_instance_norm = False,
                 use_group_norm = False,
                 use_layer_norm = False,
                 use_weight_norm = False,
                 use_dropout = False,
                 dropout_rate = 0.0,
                 ) -> None:

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self.subsample = subsample
        self.use_batch_norm = use_batch_norm
        self.use_instance_norm = use_instance_norm
        self.use_group_norm = use_group_norm
        self.use_layer_norm = use_layer_norm
        self.use_weight_norm = use_weight_norm
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        



class ConvFusion(FusionModule): 
    def __init__(self, name: str, config : ConvConfig):
        super().__init__(name)
        


class 