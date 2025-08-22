"""Some codes from https://github.com/alterzero/DBPN-Pytorch"""
import torch
import torch.nn as nn

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = nn.functional.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = nn.functional.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class ConvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='lrelu'):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        self.activation = activation
        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        out = self.conv(x)
        if self.activation is not None:
            return self.act(out)
        else:
            return out

class UpProjectionBlock(nn.Module):
    def __init__(self, num_filter, kernel_size=6, stride=2, padding=2, bias=True, activation='lrelu'):
        super(UpProjectionBlock, self).__init__()
        self.up_conv1 = PSBlock(num_filter)
        self.up_conv2 = nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, groups=1, bias=bias)
        self.r2 = nn.LeakyReLU(0.2, True)
        self.up_conv3 = PSBlock(num_filter)

    def forward(self, x):
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        l0r = self.r2(l0)
        h1 = self.up_conv3(l0r - x)
        return h1 + h0

class D_UpProjectionBlock(nn.Module):
    def __init__(self, num_filter, kernel_size=6, stride=2, padding=2, num_groups=1, bias=True, activation='lrelu'):
        super(D_UpProjectionBlock, self).__init__()
        self.conv = ConvBlock(num_filter*num_groups, num_filter, 1, 1, 0)
        self.up_conv1 = PSBlock(num_filter)
        self.up_conv2 = nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, groups=1, bias=bias)
        self.r2 = nn.LeakyReLU(0.2, True)
        self.up_conv3 = PSBlock(num_filter)

    def forward(self, x):
        x = self.conv(x)
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        l0r = self.r2(l0)
        h1 = self.up_conv3(l0r - x)
        return h1 + h0

class DownProjectionBlock(nn.Module):
    def __init__(self, num_filter, kernel_size=6, stride=2, padding=2, bias=True, activation='lrelu'):
        super(DownProjectionBlock, self).__init__()
        self.down_conv1 = nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, groups=1, bias=bias)
        self.r1 = nn.LeakyReLU(0.2, True)
        self.down_conv2 = PSBlock(num_filter)
        self.down_conv3 = nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, groups=1, bias=bias)
        self.r3 = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        l0 = self.down_conv1(x)
        l0r = self.r1(l0)
        h0 = self.down_conv2(l0r)
        l1 = self.down_conv3(h0 - x)
        l1r = self.r3(l1)
        return l1r + l0

class D_DownProjectionBlock(nn.Module):
    def __init__(self, num_filter, kernel_size=6, stride=2, padding=2, num_groups=1, bias=True, activation='lrelu'):
        super(D_DownProjectionBlock, self).__init__()
        self.conv = ConvBlock(num_filter*num_groups, num_filter, 1, 1, 0, activation=activation)
        self.down_conv1 = nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, groups=1, bias=bias)
        self.r1 = nn.LeakyReLU(0.2, True)
        self.down_conv2 = PSBlock(num_filter)
        self.down_conv3 = nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, groups=1, bias=bias)
        self.r3 = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x = self.conv(x)
        l0 = self.down_conv1(x)
        l0r = self.r1(l0)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        l1r = self.r3(l1)
        return l1r + l0


class PSBlock(nn.Module):
    def __init__(self, n_feat):
        super(PSBlock, self).__init__()
        modules = []
        modules.append(ConvBlock(n_feat, 4 * n_feat, 3, 1, 1, activation=None))
        modules.append(ChannelGate(4 * n_feat, 14, ['avg']))
        modules.append(nn.PixelShuffle(2))
        self.up = nn.Sequential(*modules)
        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        out = self.up(x)
        out = self.act(out)
        return out

class Net(nn.Module):    
    def __init__(self, opt):        

        output_size = opt.angular_out * opt.angular_out

        super(Net, self).__init__()

        # Initial Feature Extraction
        self.ps_first = nn.PixelShuffle(2)
        self.conv1 = nn.Conv2d(1, 256, 3, 1, 1)
        self.relu1 = nn.LeakyReLU(inplace=True)

        self.attSpa = SpatialGate()

        self.conv2 = nn.Conv2d(256, 128, 3, 1, 1)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.conv3 = nn.Conv2d(128, 64, 3, 1, 1)
        self.relu3 = nn.LeakyReLU(inplace=True)
        self.conv4 = nn.Conv2d(64, 49, 3, 1, 1)
        self.relu4 = nn.LeakyReLU(inplace=True)

        # Channel Fusion Module
        kernel = 6
        stride = 2
        padding = 2

        num_groups = 12
        base_filter = output_size

        self.conv5 = nn.Conv2d(49, 49, 3, 2, 1)
        self.relu5 = nn.LeakyReLU(inplace=True)
        
        self.up1 = UpProjectionBlock(base_filter, kernel, stride, padding)
        self.down1 = DownProjectionBlock(base_filter, kernel, stride, padding)
        self.up2 = UpProjectionBlock(base_filter, kernel, stride, padding)
        self.down2 = D_DownProjectionBlock(base_filter, kernel, stride, padding, 2)
        self.up3 = D_UpProjectionBlock(base_filter, kernel, stride, padding, 2)
        self.down3 = D_DownProjectionBlock(base_filter, kernel, stride, padding, 3)
        self.up4 = D_UpProjectionBlock(base_filter, kernel, stride, padding, 3)
        self.down4 = D_DownProjectionBlock(base_filter, kernel, stride, padding, 4)
        self.up5 = D_UpProjectionBlock(base_filter, kernel, stride, padding, 4)
        self.down5 = D_DownProjectionBlock(base_filter, kernel, stride, padding, 5)
        self.up6 = D_UpProjectionBlock(base_filter, kernel, stride, padding, 5)
        self.down6 = D_DownProjectionBlock(base_filter, kernel, stride, padding, 6)
        self.up7 = D_UpProjectionBlock(base_filter, kernel, stride, padding, 6)
        self.down7 = D_DownProjectionBlock(base_filter, kernel, stride, padding, 7)
        self.up8 = D_UpProjectionBlock(base_filter, kernel, stride, padding, 7)
        self.down8 = D_DownProjectionBlock(base_filter, kernel, stride, padding, 8)
        self.up9 = D_UpProjectionBlock(base_filter, kernel, stride, padding, 8)
        self.down9 = D_DownProjectionBlock(base_filter, kernel, stride, padding, 9)
        self.up10 = D_UpProjectionBlock(base_filter, kernel, stride, padding, 9)
        self.down10 = D_DownProjectionBlock(base_filter, kernel, stride, padding, 10)
        self.up11 = D_UpProjectionBlock(base_filter, kernel, stride, padding, 10)
        self.down11 = D_DownProjectionBlock(base_filter, kernel, stride, padding, 11)
        self.up12 = D_UpProjectionBlock(base_filter, kernel, stride, padding, 11)
        self.down12 = D_DownProjectionBlock(base_filter, kernel, stride, padding, 12)

        # LF Reconstruction
        self.output_conv = ConvBlock(num_groups * base_filter, output_size, 3, 1, 1, activation=None)

    def forward(self, inputs, opt):
        f1 = self.ps_first(inputs)

        c1 = self.conv1(f1)
        r1 = self.relu1(c1)

        r1 = self.attSpa(r1)

        c2 = self.conv2(r1)
        r2 = self.relu2(c2)

        c3 = self.conv3(r2)
        r3 = self.relu3(c3)

        c4 = self.conv4(r3)
        r4 = self.relu4(c4)

        c5 = self.conv5(r4)
        out0 = self.relu5(c5)

        h1 = self.up1(out0)
        l1 = self.down1(h1)
        h2 = self.up2(l1)

        concat_h = torch.cat((h2, h1), 1)
        l = self.down2(concat_h)

        concat_l = torch.cat((l, l1), 1)
        h = self.up3(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down3(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up4(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down4(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up5(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down5(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up6(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down6(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up7(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down7(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up8(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down8(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        h = self.up9(concat_l)

        concat_h = torch.cat((h, concat_h), 1)
        l = self.down9(concat_h)

        concat_l = torch.cat((l, concat_l), 1)
        x = self.output_conv(concat_l)

        x = x + out0

        return x
