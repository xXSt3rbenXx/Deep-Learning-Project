from torch import rand
import torch.nn as nn
import torch.nn.functional as F



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        # Compute left padding so conv remains causal (no lookahead)
        pad = (self.kernel_size - 1) * self.dilation

        # First causal convolution (pad only on left)
        out = F.pad(x, (pad, 0))
        out = F.relu(self.conv1(out))

        # Second causal convolution
        out = F.pad(out, (pad, 0))
        out = self.conv2(out)

        # Add residual (identity) connection
        res = x if self.downsample is None else self.downsample(x)
        return F.relu(out + res)

class TCNLayer(nn.Module):
    def __init__(self, num_inputs, hidden_dimension, blocks, kernel_size):
        """
        num_channels: list of output channels for each TCN layer/block.
        kernel_size: filter width of each 1D conv.
        """
        super().__init__()
        layers = []
        in_ch = num_inputs
        # Create a stack of ResidualBlocks with dilations 1, 2, 4...
        for i in range(blocks):
            dilation = 2 ** i      # La dilatazione consente di eseguire la convoluzione saltando i blocchi adiacenti al blocco attuale di d passi
            layers.append(ResidualBlock(in_ch, hidden_dimension, kernel_size, dilation))
            in_ch = hidden_dimension
        self.tcn = nn.Sequential(*layers)
        # Final linear layer to produce a single output value
        

    def forward(self, x, adj):
        """
        x has shape (batch_size, channels, seq_len).
        """
        return self.tcn(x)            # (batch, channels, seq_len)
              