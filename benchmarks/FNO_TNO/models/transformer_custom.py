import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from utils import reflect_pad


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()

        self.scale = nn.Parameter(torch.sqrt(torch.FloatTensor([d_k])), requires_grad=False)
        self.dropout = nn.Dropout(dropout)

    def custom_softmax(self, x, coords=None, dim=-1):

        exp_x = torch.exp(x - x.max(dim=dim, keepdim=True)[0])
        #reweighting of exp_x along seq_len dimension
        if coords is not None:
            softmax_x = exp_x / (0.5*coords*(exp_x[...,1:]+exp_x[...,:-1])).sum(dim=dim, keepdim=True)
        else:
            softmax_x = exp_x / exp_x.sum(dim=dim, keepdim=True)
        return softmax_x

    def forward(self, query, key, value, coords, key_padding_mask=None):
        # Custom logic for attention calculation

        scores = torch.einsum("bhld,bhsd->bhls", query, key) / self.scale

        #makes sure that if domain_dim is not 1, then coords is handled differently
        if coords.shape[1]==1:
            coords = torch.abs(coords[1:,...] - coords[:-1,...])
            coords = coords.permute(1,2,0).unsqueeze(0)
            #coords is now a vector of distances between coordinates such that the shape is broadcastable with scores[...,1:] (1,1,1,seq_len-1)
        else:
            coords = None

        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attention_weights = self.custom_softmax(scores, coords=coords, dim=-1)
        attention_weights = self.dropout(attention_weights)

        #reweighting of value along seq_len dimension
        if coords is not None:
            value1 = coords.permute(0,1,3,2)*value[...,1:,:]
            output1 = torch.einsum("bhls,bhsd->bhld", attention_weights[...,1:], value1)
            value2 = coords.permute(0,1,3,2)*value[...,:-1,:]
            output2 = torch.einsum("bhls,bhsd->bhld", attention_weights[...,:-1], value2)
            output = 0.5*(output1 + output2)
        else:
            output = torch.einsum("bhls,bhsd->bhld", attention_weights, value)

        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.W_q = nn.Linear(d_model, nhead*self.d_k)
        self.W_k = nn.Linear(d_model, nhead*self.d_k)
        self.W_v = nn.Linear(d_model, nhead*self.d_k)
        self.scaled_dot_product_attention = ScaledDotProductAttention(self.d_k, dropout=dropout)
        self.W_o = nn.Linear(nhead*self.d_k, d_model)

    def split_heads(self, x):
        #batch_size, seq_length, d_model = x.size()
        batch_size = x.shape[0]
        return x.view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        #batch_size, nhead, seq_length, d_k = x.size()
        batch_size = x.shape[0]
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.nhead * self.d_k)

    def forward(self, x, coords_x, mask=None):

        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))

        attn_output = self.scaled_dot_product_attention(Q, K, V, coords_x, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, activation):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.activation = getattr(F, activation)

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, activation="relu", norm_first=True, do_layer_norm=True, dim_feedforward=2048, batch_first=True):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.feed_forward = FeedForward(d_model, dim_feedforward, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, coords_x, mask=None):
        attn_output = self.self_attn(x, coords_x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers=1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, x, coords_x, mask=None):
        for layer in self.layers:
            x = layer(x, coords_x, mask=mask)
        return x

###############################################################################################################
###############################################################################################################
#SpectralConv2d Module
###############################################################################################################
###############################################################################################################


class SpectralConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels))
        #use torch.view_as_real if using ddp
        self.weights1 = nn.Parameter(self.scale * torch.view_as_real(torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)))
        self.weights2 = nn.Parameter(self.scale * torch.view_as_real(torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)))
        #self.weights1 = nn.Parameter(self.scale * (torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)))
        #self.weights2 = nn.Parameter(self.scale * (torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)))
    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)
    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        #use torch.view_as_complex if using ddp
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], torch.view_as_complex(self.weights1))
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], torch.view_as_complex(self.weights2))
        #out_ft[:, :, :self.modes1, :self.modes2] = \
            #self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        #out_ft[:, :, -self.modes1:, :self.modes2] = \
            #self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


###############################################################################################################
###############################################################################################################
#MODULES FOR VANILLA TRANSFORMER OPERATOR ARCHITECTURE
###############################################################################################################
###############################################################################################################


class MultiheadAttention_Operator(nn.Module):
    def __init__(self, d_model, nhead, modes1, modes2, im_size, dropout=0.1):
        super(MultiheadAttention_Operator, self).__init__()
        self.nhead = nhead

        self.query_operator = SpectralConv2d(d_model, d_model, modes1, modes2, nhead)
        self.key_operator = SpectralConv2d(d_model, d_model, modes1, modes2, nhead)
        self.value_operator = SpectralConv2d(d_model, d_model, modes1, modes2, nhead)

        self.scaled_dot_product_attention = ScaledDotProductAttention_Operator(d_model, im_size, dropout=dropout)

        self.out_linear = nn.Linear(d_model*nhead, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):

        batch, num_patches, patch_size, patch_size, d_model = x.size()
        query = self.query_operator(x).permute(0,5,1,2,3,4)
        key = self.key_operator(x).permute(0,5,1,2,3,4)
        value = self.value_operator(x).permute(0,5,1,2,3,4)
        # query, key, and value are of shape (batch, nhead, num_patches,x,y , d_model)

        # Scaled Dot Product Attention
        attention_output, _ = self.scaled_dot_product_attention(query, key, value, key_padding_mask=key_padding_mask)

        # Reshape and linear transformation
        attention_output = attention_output.reshape(batch, num_patches, patch_size, patch_size,-1)
        output = self.out_linear(attention_output)
        output = self.dropout(output)

        return output

class ScaledDotProductAttention_Operator(nn.Module):
    def __init__(self, d_model, im_size, dropout=0.1):
        super(ScaledDotProductAttention_Operator, self).__init__()
        #d_model* or just d_model?
        self.scale = nn.Parameter(torch.sqrt(torch.FloatTensor([((im_size)**4)])), requires_grad=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None):
        # Custom logic for attention calculation
        scores = torch.einsum("bnpxyd,bnqxyd->bnpq", query, key) / self.scale

        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.einsum("bnpq,bnqxyd->bnpxyd", attention_weights, value)
        output = output.permute(0,2,3,4,5,1)

        return output, attention_weights

class TransformerEncoderLayer_Operator(nn.Module):#
    def __init__(self, d_model, nhead, dropout=0.1, activation="relu", norm_first=True, do_layer_norm=True, dim_feedforward=2048, modes=None, patch_size=1, im_size=64, batch_first=True):
        super(TransformerEncoderLayer_Operator, self).__init__()

        # Self-attention layer
        if modes is None:
            modes2 = patch_size//2+1
            modes1 = patch_size//2+1
        else:
            modes1 = modes[0]
            modes2 = modes[1]
        #or im_size?
        self.self_attn = MultiheadAttention_Operator(d_model, nhead, modes1, modes2, im_size, dropout=dropout)

        # Feedforward layer
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Activation function
        self.activation = getattr(F, activation)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Whether to normalize before the self-attention layer
        self.norm_first = norm_first

        # Whether to apply layer normalization at the end
        self.do_layer_norm = do_layer_norm

    def forward(self, x, mask=None):
        if self.norm_first:
            x = self.norm1(x)

        # Self-attention
        attn_output = self.self_attn(x, key_padding_mask=mask)

        # Residual connection and layer normalization
        x = x + self.dropout(attn_output)
        if not self.norm_first:
            x = self.norm1(x)

        # Feedforward layer
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))

        # Residual connection and layer normalization
        x = x + self.dropout(ff_output)
        if self.do_layer_norm:
            x = self.norm2(x)

        return x

class TransformerEncoder_Operator(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder_Operator, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x)
        return x


###############################################################################################################
###############################################################################################################
#MODULES FOR ViT OPERATOR ARCHITECTURE
###############################################################################################################
###############################################################################################################


class SpectralConv2d_in(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_in, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        #use torch.view_as_real if using ddp
        #self.weights1 = nn.Parameter(self.scale * torch.view_as_real(torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)))
        #self.weights2 = nn.Parameter(self.scale * torch.view_as_real(torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)))
        self.weights1 = nn.Parameter(self.scale * (torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)))
        self.weights2 = nn.Parameter(self.scale * (torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch,num_patches, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, num_patches, out_channel, x,y)
        return torch.einsum("bnixy,ioxy->bnoxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        num_patches = x.shape[1]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)
        max_modes2 = min(self.modes2, x_ft.shape[-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, num_patches, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        #use torch.view_as_complex if using ddp
        #out_ft[:,:, :, :self.modes1, :max_modes2] = \
            #self.compl_mul2d(x_ft[:, :,:, :self.modes1, :max_modes2], torch.view_as_complex(self.weights1)[..., :max_modes2])
        #out_ft[:,:, :, -self.modes1:, :self.modes2] = \
            #self.compl_mul2d(x_ft[:, :,:, -self.modes1:, :max_modes2], torch.view_as_complex(self.weights2)[..., :max_modes2])
        out_ft[:,:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :,:, :self.modes1, :max_modes2], self.weights1[..., :max_modes2])
        out_ft[:,:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :,:, -self.modes1:, :max_modes2], self.weights2[..., :max_modes2])

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class MultiheadAttention_ViTNO(nn.Module):
    def __init__(self, d_model, nhead, im_size, dropout=0.1):
        super(MultiheadAttention_ViTNO, self).__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.W_q = nn.Linear(d_model, nhead*self.d_k)
        self.W_k = nn.Linear(d_model, nhead*self.d_k)
        self.W_v = nn.Linear(d_model, nhead*self.d_k)
        self.scaled_dot_product_attention = ScaledDotProductAttention_ViTNO(self.d_k, im_size, dropout=dropout)
        self.W_o = nn.Linear(nhead*self.d_k, d_model)

    def split_heads(self, x):
        batch, num_patches, patch_size, patch_size, nhead_times_d_K = x.size()
        return x.view(batch, num_patches, patch_size, patch_size, self.d_k, self.nhead).permute(0,5,1,2,3,4)

    def combine_heads(self, x):
        batch, num_patches, patch_size, patch_size, d_K, num_heads = x.size()
        return x.reshape(batch, num_patches, patch_size, patch_size, self.nhead*self.d_k)

    def forward(self, x, key_padding_mask=None):

        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))
        # query, key, and value are of shape (batch, nhead, num_patches,x,y , d_K)

        attn_output = self.scaled_dot_product_attention(Q, K, V, key_padding_mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

class ScaledDotProductAttention_ViTNO(nn.Module):
    def __init__(self, d_k, im_size, dropout=0.1):
        super(ScaledDotProductAttention_ViTNO, self).__init__()
        #d_K* or just d_K?
        self.scale = nn.Parameter(torch.sqrt(torch.FloatTensor([((im_size)**4)])), requires_grad=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None):
        # Custom logic for attention calculation
        scores = torch.einsum("bnpxyd,bnqxyd->bnpq", query, key) / self.scale

        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.einsum("bnpq,bnqxyd->bnpxyd", attention_weights, value)
        output = output.permute(0,2,3,4,5,1)

        return output

class TransformerEncoderLayer_ViTNO(nn.Module):#

    def __init__(self, d_model, nhead, dropout=0.1, activation="relu", norm_first=True, do_layer_norm=True, dim_feedforward=2048, modes=None, patch_size=1, im_size=64, batch_first=True):
        super(TransformerEncoderLayer_ViTNO, self).__init__()

        # Self-attention layer
        if modes is None:
            modes2 = patch_size//2+1
            modes1 = patch_size//2+1
        else:
            modes1 = modes[0]
            modes2 = modes[1]
        #or im_size?
        self.self_attn = MultiheadAttention_ViTNO(d_model, nhead, im_size, dropout=dropout)

        # Feedforward layer
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Activation function
        self.activation = getattr(F, activation)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Whether to normalize before the self-attention layer
        self.norm_first = norm_first

        # Whether to apply layer normalization at the end
        self.do_layer_norm = do_layer_norm


    def forward(self, x, mask=None):

        if self.norm_first:
            x = self.norm1(x)

        # Self-attention
        attn_output = self.self_attn(x, key_padding_mask=mask)

        # Residual connection and layer normalization
        x = x + self.dropout(attn_output)
        if not self.norm_first:
            x = self.norm1(x)

        # Feedforward layer
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))

        # Residual connection and layer normalization
        x = x + self.dropout(ff_output)
        if self.do_layer_norm:
            x = self.norm2(x)

        return x


###############################################################################################################
###############################################################################################################
#MODULES FOR INTEGRAL QKV TRANSFORMER ARCHITECTURE
###############################################################################################################
###############################################################################################################


class SpectralConv2d_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, nhead):
        super(SpectralConv2d_Attention, self).__init__()
        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.nhead = nhead
        self.scale = (1 / (in_channels * out_channels))
        #use torch.view_as_real if using ddp
        self.weights1 = nn.Parameter(self.scale * torch.view_as_real(torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.nhead, dtype=torch.cfloat)))
        self.weights2 = nn.Parameter(self.scale * torch.view_as_real(torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.nhead, dtype=torch.cfloat)))
        #self.weights1 = nn.Parameter(self.scale * (torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.nhead, dtype=torch.cfloat)))
        #self.weights2 = nn.Parameter(self.scale * (torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.nhead, dtype=torch.cfloat)))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, num_patches, in_channel, x,y ), (in_channel, out_channel, x,y, nhead) -> (batch, num_patches, out_channel, x,y, nhead)
        return torch.einsum("bnixy,ioxyh->bnoxyh", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        num_patches = x.shape[1]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        #####
        x = torch.permute(x, (0,1,4,2,3))
        #x is of shape (batch, num_patches, d_model, x, y)
        x_ft = torch.fft.rfft2(x)
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, num_patches, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, self.nhead, dtype=torch.cfloat, device=x.device)
        #use torch.view_as_complex if using ddp
        out_ft[:, :, :, :self.modes1, :self.modes2, :] = \
            self.compl_mul2d(x_ft[:, :, :, :self.modes1, :self.modes2], torch.view_as_complex(self.weights1))
        out_ft[:, :, :, -self.modes1:, :self.modes2, :] = \
            self.compl_mul2d(x_ft[:, :, :, -self.modes1:, :self.modes2], torch.view_as_complex(self.weights2))
        #out_ft[:, :, :, :self.modes1, :self.modes2, :] = \
            #self.compl_mul2d(x_ft[:, :, :, :self.modes1, :self.modes2], self.weights1)
        #out_ft[:, :, :, -self.modes1:, :self.modes2, :] = \
            #self.compl_mul2d(x_ft[:, :, :, -self.modes1:, :self.modes2], self.weights2)
        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), dim=(-3, -2))
        #####
        x = torch.permute(x, (0,1,3,4,2,5))
        #x is of shape (batch, num_patches, x, y, d_model, nhead)
        return x

class MultiheadAttention_Conv(nn.Module):

    def __init__(self, d_model, nhead, modes1, modes2, im_size, sample_rate, dropout=0.1):
        super(MultiheadAttention_Conv, self).__init__()
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.im_size = im_size
        self.sample_rate = sample_rate


        self.query_operator = SpectralConv2d_Attention(d_model, self.d_k, modes1, modes2, nhead)
        self.key_operator = SpectralConv2d_Attention(d_model, self.d_k, modes1, modes2, nhead)
        self.value_operator = SpectralConv2d_Attention(d_model, self.d_k, modes1, modes2, nhead)

        self.scaled_dot_product_attention = ScaledDotProductAttention_Conv(d_model, im_size, dropout=dropout)

        self.out_linear = nn.Linear(nhead*self.d_k, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        batch, num_patches, patch_size, patch_size, d_model = x.size()

        ###should write a split heads function to do the permutation
        query = self.query_operator(x).permute(0,5,1,2,3,4)
        key = self.key_operator(x).permute(0,5,1,2,3,4)
        value = self.value_operator(x).permute(0,5,1,2,3,4)
        
        # query, key, and value are of shape (batch, nhead, num_patches,x,y , d_model)

        # Scaled Dot Product Attention

        attention_output, _ = self.scaled_dot_product_attention(query, key, value, key_padding_mask=key_padding_mask)

        ######
        ###this should be the combine heads function
        attention_output = attention_output.reshape(batch, num_patches, patch_size, patch_size,-1)
        #####
        output = self.out_linear(attention_output)
        output = self.dropout(output)
        return output

class ScaledDotProductAttention_Conv(nn.Module):

    def __init__(self, d_model, im_size, dropout=0.1):
        super(ScaledDotProductAttention_Conv, self).__init__()
        #d_model* or just d_model?
        self.scale = nn.Parameter(torch.sqrt(torch.FloatTensor([((im_size)**4)])), requires_grad=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None):
        # Custom logic for attention calculation
        scores = torch.einsum("bnpxyd,bnqxyd->bnpq", query, key) / self.scale
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.einsum("bnpq,bnqxyd->bnpxyd", attention_weights, value)
        output = output.permute(0,2,3,4,5,1)
        return output, attention_weights

class TransformerEncoderLayer_Conv(nn.Module):#

    def __init__(self, d_model, nhead, dropout=0.1, activation="relu", norm_first=True, do_layer_norm=True, dim_feedforward=2048, modes=None, patch_size=1, im_size=64, sample_rate=1, batch_first=True):
        super(TransformerEncoderLayer_Conv, self).__init__()
        # Self-attention layer
        if modes is None:
            modes2 = patch_size//2+1
            modes1 = patch_size//2+1
        else:
            modes1 = modes[0]
            modes2 = modes[1]

        self.patch_size = patch_size
        self.im_size = im_size
        self.size_row = im_size
        self.size_col = im_size
        self.d_model = d_model

        #or im_size?
        self.self_attn = MultiheadAttention_Conv(d_model, nhead, modes1, modes2, im_size, sample_rate, dropout=dropout)
        # Feedforward layer
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Activation function

        self.activation = getattr(F, activation)
        # Dropout
        self.dropout = nn.Dropout(dropout)
        # Whether to normalize before the self-attention layer
        self.norm_first = norm_first
        # Whether to apply layer normalization at the end
        self.do_layer_norm = do_layer_norm


    def forward(self, x, mask=None):
        if self.norm_first:
            x = self.norm1(x)
        # Self-attention
        attn_output = self.self_attn(x, key_padding_mask=mask)
        # Residual connection and layer normalization
        x = x + self.dropout(attn_output)
        if not self.norm_first:
            x = self.norm1(x)
        # Feedforward layer
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        # Residual connection and layer normalization
        x = x + self.dropout(ff_output)
        if self.do_layer_norm:
            x = self.norm2(x)

        return x


###############################################################################################################
###############################################################################################################
#MODULES FOR INTEGRAL QKV TRANSFORMER ARCHITECTURE WITH FOLDING
###############################################################################################################
###############################################################################################################


class MultiheadAttention_Conv_Reflect(nn.Module):

    def __init__(self, d_model, nhead, modes1, modes2, im_size, sample_rate, dropout=0.1):
        super(MultiheadAttention_Conv_Reflect, self).__init__()
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.im_size = im_size
        self.sample_rate = sample_rate


        self.query_operator = SpectralConv2d_Attention(d_model, self.d_k, modes1, modes2, nhead)
        self.key_operator = SpectralConv2d_Attention(d_model, self.d_k, modes1, modes2, nhead)
        self.value_operator = SpectralConv2d_Attention(d_model, self.d_k, modes1, modes2, nhead)

        self.scaled_dot_product_attention = ScaledDotProductAttention_Conv(d_model, im_size, dropout=dropout)

        self.out_linear = nn.Linear(nhead*self.d_k, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        batch, num_patches, patch_size, patch_size, d_model = x.size()

        ###
        padding = patch_size-1
        x = x.permute(0,1,4,2,3)
        # Reshape to merge batch and num_patches for isolated spatial padding
        x = x.reshape(-1, d_model, patch_size, patch_size)  # Shape: (batch * num_patches, d_model, patch_size, patch_size)
        #x = F.pad(x, (padding,padding,padding,padding), 'constant', 0)
        x = reflect_pad(x, (0, padding, 0, 0))
        x = reflect_pad(x, (0, 0, 0, padding))
        # Reshape back to original structure
        x = x.reshape(batch, num_patches, d_model, 2*patch_size-1, 2*patch_size-1)
        x = x.permute(0,1,3,4,2)


        ###should write a split heads function to do the permutation
        query = self.query_operator(x).permute(0,5,1,2,3,4)
        key = self.key_operator(x).permute(0,5,1,2,3,4)
        value = self.value_operator(x).permute(0,5,1,2,3,4)
        
       
        # query, key, and value are of shape (batch, nhead, num_patches,x,y , d_model)
        start_idx = 0
        end_idx = patch_size

        query = query[:, :, :, start_idx:end_idx, start_idx:end_idx, :]
        key = key[:, :, :, start_idx:end_idx, start_idx:end_idx, :]
        value = value[:, :, :, start_idx:end_idx, start_idx:end_idx, :]
        
        # Scaled Dot Product Attention

        attention_output, _ = self.scaled_dot_product_attention(query, key, value, key_padding_mask=key_padding_mask)

        ######
        ###this should be the combine heads function
        attention_output = attention_output.reshape(batch, num_patches, patch_size, patch_size,-1)
        #####
        output = self.out_linear(attention_output)
        output = self.dropout(output)
        return output

class TransformerEncoderLayer_Conv_Reflect(nn.Module):#

    def __init__(self, d_model, nhead, dropout=0.1, activation="relu", norm_first=True, do_layer_norm=True, dim_feedforward=2048, modes=None, patch_size=1, im_size=64, sample_rate=1, batch_first=True):
        super(TransformerEncoderLayer_Conv_Reflect, self).__init__()
        # Self-attention layer
        if modes is None:
            modes2 = patch_size//2+1
            modes1 = patch_size//2+1
        else:
            modes1 = modes[0]
            modes2 = modes[1]

        self.patch_size = patch_size
        self.im_size = im_size
        self.size_row = im_size
        self.size_col = im_size
        self.d_model = d_model

        #or im_size?
        self.self_attn = MultiheadAttention_Conv_Reflect(d_model, nhead, modes1, modes2, im_size, sample_rate, dropout=dropout)
        # Feedforward layer
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Activation function

        self.activation = getattr(F, activation)
        # Dropout
        self.dropout = nn.Dropout(dropout)
        # Whether to normalize before the self-attention layer
        self.norm_first = norm_first
        # Whether to apply layer normalization at the end
        self.do_layer_norm = do_layer_norm


    def forward(self, x, mask=None):
        if self.norm_first:
            x = self.norm1(x)
        # Self-attention
        attn_output = self.self_attn(x, key_padding_mask=mask)
        # Residual connection and layer normalization
        x = x + self.dropout(attn_output)
        if not self.norm_first:
            x = self.norm1(x)
        # Feedforward layer
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        # Residual connection and layer normalization
        x = x + self.dropout(ff_output)
        if self.do_layer_norm:
            x = self.norm2(x)

        return x

######################################################################################
######################################################################################
######################################################################################
######################################################################################

######################################################################################
######################################################################################
######################################################################################
######################################################################################

#remove nn.Module and self.scale/decay

class Smoothing2d():

    def __init__(self):
        super(Smoothing2d, self).__init__()
        #self.scale = nn.Parameter(torch.tensor(1.),requires_grad=True)
        #self.decay = nn.Parameter(torch.tensor(1.),requires_grad=True)
        #self.linear1 = nn.Linear(128,128)
        #self.relu = nn.ReLU()
        #self.linear2 = nn.Linear(128,2)
        #self.softplus = nn.Softplus()

    def kernel_ft(self, x, coords_x):
        N, rows, cols = x.shape
        #import pdb; pdb.set_trace()
        #coords = coords_x[0,...].permute(0,2,1)*(rows-1)+1
        #coords = torch.sum(coords**2, dim=0)[:, :cols//2+1]

        freq_list1 = torch.cat((torch.arange(start=0, end=rows//2, step=1),\
                                torch.arange(start=-rows//2, end=0, step=1)), 0)
        k1 = freq_list1.view(-1,1).repeat(1, rows).to(x)

        freq_list2 = torch.cat((torch.arange(start=0, end=cols//2, step=1),\
                                torch.arange(start=-cols//2, end=0, step=1)), 0)

        k2 = freq_list2.view(1,-1).repeat(cols, 1).to(x)

        coords = k1**2 + k2**2
        coords = coords[:, :cols//2+1]
    
        return coords


    #the following is learning the scale and decay
    def __call__(self, x, coords_x):

        x_ft = torch.fft.rfft2(x)
        #params = self.linear2(self.relu(self.linear1(torch.ones(128).to(x))))
        #scale, decay = params[0], params[1]
        #scale = self.softplus(self.scale)
        #decay = self.softplus(self.decay)
        scale = 0.001
        decay = 1.001
        #* (x.shape[1]**2)
        #x_ft = x_ft *(1/((1+scale*4*(np.pi**2)*self.kernel_ft(x,coords_x))**decay))
        x_ft = x_ft *(1/((1+scale*self.kernel_ft(x,coords_x))**decay))
        x = torch.fft.irfft2(x_ft, s=(x.size(-2), x.size(-1)))

        return x
    '''
    #the following is learning the coefficients completely
    def forward(self, x, coords_x):
        x_ft = torch.fft.rfft2(x)
        params = self.linear2(self.relu(self.linear1(torch.ones(128).to(x))))
        scale, decay = params[0], params[1]
        x_ft = x_ft * (1/((1+(scale**2)*(4*(np.pi**2)/(x.shape[1]**2))*self.kernel_ft(x,coords_x))**decay))
        x = torch.fft.irfft2(x_ft, s=(x.size(-2), x.size(-1)))
        return x
    '''

class Smoothing2D_new:
    def __init__(self, epsilon=1e-6, bc_type="periodic"):
        """
        Parameters
        ----------
        epsilon : float
            Regularization to avoid division by zero for k=0 mode.
        bc_type : str
            "periodic" for [0, 2pi]^2 periodic BCs
            "dirichlet" for [0, 1]^2 zero Dirichlet BCs
        """
        assert bc_type in ("periodic", "dirichlet")
        self.epsilon = epsilon
        self.bc_type = bc_type
        self._cache = {}

    # ==============================
    # Periodic kernel: integer wavenumbers
    # ==============================
    def _periodic_kernel(self, H, W, device, dtype):
        key = ("per", H, W, device, dtype)
        if key in self._cache:
            return self._cache[key]

        # integer wavenumbers for [0, 2pi]
        kx = torch.fft.fftfreq(H, d=1.0) * H  # shape (H,)
        ky = torch.fft.rfftfreq(W, d=1.0) * W  # shape (W//2+1,)
        KX, KY = torch.meshgrid(kx, ky, indexing='ij')
        k2 = KX**2 + KY**2

        self._cache[key] = k2.to(device=device, dtype=dtype)
        return self._cache[key]

    # ==============================
    # Dirichlet kernel: sine transform eigenvalues
    # ==============================
    def _dirichlet_kernel(self, H, W, device, dtype):
        key = ("dir", H, W, device, dtype)
        if key in self._cache:
            return self._cache[key]

        # Eigenvalues for discrete Laplacian with 0 Dirichlet BCs on [0,1]
        # Indices start at 1 for sine modes
        m = torch.arange(1, H+1, device=device, dtype=dtype).view(-1, 1)
        n = torch.arange(1, W+1, device=device, dtype=dtype).view(1, -1)
        # Continuous Laplacian eigenvalues: (m*pi)^2 + (n*pi)^2
        lam = (m * torch.pi)**2 + (n * torch.pi)**2

        self._cache[key] = lam
        return lam

    # ==============================
    # Apply smoothing: inverse(-Laplace)
    # ==============================
    def __call__(self, x):
        """
        Smooth x by applying ( -Δ )^{-1} in spectral space.
        """
        H, W = x.shape[-2:]
        device, dtype = x.device, x.dtype

        if self.bc_type == "periodic":
            k2 = self._periodic_kernel(H, W, device, dtype)
            x_ft = torch.fft.rfft2(x)
            inv_k2 = 1.0 / (k2 + self.epsilon)
            y_ft = x_ft * inv_k2
            return torch.fft.irfft2(y_ft, s=(H, W))

        elif self.bc_type == "dirichlet":
            # Use DST via DCT trick (PyTorch doesn't have DST directly)
            # We'll implement a DST-I using FFT
            def dst2(u):
                # DST-I on each axis
                U = torch.fft.fft(torch.cat(
                    [torch.zeros_like(u[:1, :]), u, torch.zeros_like(u[:1, :]), -u.flip(0)], dim=0
                ), dim=0).imag[1:H+1, :]
                U = torch.fft.fft(torch.cat(
                    [torch.zeros_like(U[:, :1]), U, torch.zeros_like(U[:, :1]), -U.flip(1)], dim=1
                ), dim=1).imag[:, 1:W+1]
                return U

            def idst2(U):
                # Inverse DST-I
                u = torch.fft.ifft(torch.cat(
                    [torch.zeros_like(U[:1, :]), U, torch.zeros_like(U[:1, :]), -U.flip(0)], dim=0
                ), dim=0).imag[1:H+1, :] * (2 / (H+1))
                u = torch.fft.ifft(torch.cat(
                    [torch.zeros_like(u[:, :1]), u, torch.zeros_like(u[:, :1]), -u.flip(1)], dim=1
                ), dim=1).imag[:, 1:W+1] * (2 / (W+1))
                return u

            lam = self._dirichlet_kernel(H, W, device, dtype)
            U = dst2(x)
            U /= (lam + self.epsilon)
            return idst2(U)