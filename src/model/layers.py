import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, **kwargs)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UnetUp(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, fusion='concat',  skip_channels=None, **kwargs):
        super().__init__()
        self.fusion = fusion
        self.up_layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.skip_layer = nn.Sequential(
            nn.Conv2d(in_channels // 2 if fusion == 'add' else skip_channels, in_channels if fusion == 'add' else skip_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels if fusion == 'add' else skip_channels),
            nn.LeakyReLU(inplace=True)
        )
        
        self.conv = DoubleConv(in_channels + skip_channels if self.fusion =="concat" else in_channels, out_channels, **kwargs)
        
    def forward(self, enc_feat, skip_feat):
        enc_feat = self.up_layer(enc_feat)
        skip_feat = self.skip_layer(skip_feat)
        
        if self.fusion == 'concat':
            x = torch.cat([skip_feat, enc_feat], dim=1)
        elif self.fusion == 'add':
            x = skip_feat + enc_feat
        return self.conv(x)

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=kernel_size // 2) #4 times because of 3 gates + cell
        self.hidden_dim = hidden_dim

    def forward(self, x, hidden):
        h_cur, c_cur = hidden
        combined = torch.cat([x, h_cur], dim=1)
        conv_out = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_out, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, spatial_size, device):
        height, width = spatial_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device))
    
class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)

    def forward(self, x):
        b, t, c, h, w = x.size() # x: [B, T, C, H, W]
        h_t, c_t = self.cell.init_hidden(b, (h, w), x.device)
        for i in range(t): #forward cell over all time steps by updating hidden states
            h_t, c_t = self.cell(x[:, i, :, :, :], (h_t, c_t))
        return h_t  # return last hidden state