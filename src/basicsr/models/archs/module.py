import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange


class EX(nn.Module):
    def __init__(self, dim):
        super(EX, self).__init__()

        in_channels = dim
        out_channels = dim

        # self.convs = nn.ModuleList([
        #     nn.Conv2d(in_channels, out_channels, kernel_size=1),  # standard 1x1
        #     nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # standard 3x3
        #     nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),  # standard 5x5
        #     #nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3),  # standard 7x7
        #     nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=in_channels),  # depthwise 1x1
        #     nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=in_channels),  # depthwise 3x3
        #     nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, groups=in_channels),  # depthwise 5x5
        #     #nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3, groups=in_channels),  # depthwise 7x7
        # ])

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1),  # standard 1x1
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # standard 3x3
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),  # standard 5x5
            nn.Conv2d(in_channels, out_channels, kernel_size=1),  # standard 1x1
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # standard 3x3
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),  # standard 5x5


            nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=in_channels),  # depthwise 1x1
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=in_channels),  # depthwise 3x3
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, groups=in_channels),  # depthwise 5x5
            nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=in_channels),  # depthwise 1x1
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=in_channels),  # depthwise 3x3
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, groups=in_channels),  # depthwise 5x5


        ])

        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 12))
        self.topk = 4
        self.output = nn.Conv2d(in_channels*self.topk, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, weights):


        weights = self.adaptive_pool(weights)
        weights = torch.squeeze(weights, dim=1)
        weights = torch.squeeze(weights, dim=1)
        weights = F.softmax(weights, dim=-1)


        _, topk_indices = weights.topk(self.topk, dim=1)


        outputs = []
        for i in range(self.topk):
            conv_idx = topk_indices[0][i]
            conv = self.convs[conv_idx]
            outputs.append(conv(x))


        concat_output = torch.cat(outputs, dim=1)


        concat_output = self.output(concat_output)


        return concat_output


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class Attention_CA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_CA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, Q,kv):

        x = Q
        b, c, h, w = x.shape
        _, _, l = kv.shape
        x_pool = F.avg_pool1d(kv.transpose(1, 2), 4).transpose(1, 2)
        k =  x_pool
        v = x_pool
        q = x
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) l -> b head c l', head=self.num_heads)
        v = rearrange(v, 'b (head c) l -> b head c l', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        v = torch.nn.functional.normalize(v, dim=-1)

        if h == 512:
            k =k.repeat(1, 1, 1, 64)
            v = v.repeat(1, 1, 1, 64)
        elif h == 256:
            k =k.repeat(1, 1, 1, 16)
            v = v.repeat(1, 1, 1, 16)
        elif h == 128:
            k =k.repeat(1, 1, 2, 4)
            v = v.repeat(1, 1, 2, 4)
        else:
            k =k.repeat(1, 1, 4, 1)
            v = v.repeat(1, 1, 4, 1)


        kt = k.transpose(-1, -2)
        attn = q @ kt
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)


        return out


class Attention_C(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_C, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q_dwconv = nn.Conv2d(dim * 1, dim * 1, kernel_size=3, stride=1, padding=1, groups=dim * 1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.ca1 =  Attention_CA(dim=dim,num_heads=num_heads,bias=False)
        self.ex2 = EX(dim=dim)

    def forward(self, I,T):

        x = I
        b, c, h, w = x.shape
        F = self.ca1(I,T)
        F = self.ex2(I, F)
        kv = self.kv_dwconv(self.kv(F))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(F)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

class MEAB(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(MEAB, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention_C(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x,w):
        x = x + self.attn(self.norm1(x),w)
        x = x + self.ffn(self.norm2(x))

        return x


