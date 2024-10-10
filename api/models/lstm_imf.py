import torch
import torch.nn as nn
from RevIN import RevIN
import logging
from math import sqrt

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim_in, dim_k, dim_v, num_heads=16):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)

    def forward(self, x):
        B, N, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.dim_k // nh
        dv = self.dim_v // nh

        q = self.linear_q(x).reshape(B, N, nh, dk).transpose(1, 2)
        k = self.linear_k(x).reshape(B, N, nh, dk).transpose(1, 2)
        v = self.linear_v(x).reshape(B, N, nh, dv).transpose(1, 2)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact
        dist = torch.softmax(dist, dim=-1)

        att = torch.matmul(dist, v)
        att = att.transpose(1, 2).reshape(B, N, self.dim_v)
        
        return att

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.seg_len = configs.seg_len
        self.dec_way = configs.dec_way
        self.channel_id = configs.channel_id
        self.revin = configs.revin

        self.seg_num_x = self.seq_len // self.seg_len
        self.seg_num_y = self.pred_len // self.seg_len

        logger.info(f"Initializing Model with d_model={self.d_model}, dec_way={self.dec_way}")

        self.value_embedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=2,
            bias=True,
            batch_first=True,
            bidirectional=False
        )

        self.attention = MultiHeadSelfAttention(dim_in=self.d_model, dim_k=self.d_model, dim_v=self.d_model, num_heads=4)

        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.seg_len)
        )

        if self.dec_way == "pmf":
            if self.channel_id:
                self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.d_model // 2))
                self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))
            else:
                self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.d_model))

        if self.revin:
            self.revin_layer = RevIN(self.enc_in, affine=False, subtract_last=False)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.normalization_with_permute(x)

        x = self.segment_embed(x)

        hn, cn = self.encode(x)

        hn = self.apply_attention(hn)

        y = self.decode(hn, cn, batch_size)

        y = self.permute_and_denormalize(y)

        return y

    def normalization_with_permute(self, x):
        if self.revin:
            x = self.revin_layer(x, 'norm')
            return x.permute(0, 2, 1)
        else:
            self.seq_last = x[:, -1:, :].detach()
            return (x - self.seq_last).permute(0, 2, 1)

    def segment_embed(self, x):
        x = x.reshape(-1, self.seg_num_x, self.seg_len)
        return self.value_embedding(x)

    def encode(self, x):
        _, (hn, cn) = self.lstm(x)
        return hn, cn

    def apply_attention(self, hn):
        hn = hn.permute(1, 0, 2)
        hn = self.attention(hn)
        return hn.permute(1, 0, 2)

    def decode(self, hn, cn, batch_size):
        if self.dec_way == "rmf":
            return self.recurrent_multi_step_forecasting_decode(hn, cn)
        elif self.dec_way == "pmf":
            return self.parallel_multi_step_forecasting_decode(hn, cn, batch_size)

    def recurrent_multi_step_forecasting_decode(self, hn, cn):
        y = []
        for i in range(self.seg_num_y):
            yy = self.predict(hn).permute(1, 0, 2)
            y.append(yy)
            yy = self.value_embedding(yy)
            _, (hn, cn) = self.lstm(yy, (hn, cn))
        output = torch.stack(y, dim=1).squeeze(2).reshape(-1, self.enc_in, self.pred_len)

        return output

    # pmf with attention
    def parallel_multi_step_forecasting_decode(self, hn, cn, batch_size):
        pos_emb = self.position_embed_get(batch_size)
        
        # Adjust for multiple layers
        num_layers = self.lstm.num_layers
        
        # Reshape hn and cn to account for multiple layers
        hn_repeat = hn.repeat(1, 1, self.seg_num_y).view(num_layers, -1, self.d_model)
        cn_repeat = cn.repeat(1, 1, self.seg_num_y).view(num_layers, -1, self.d_model)

        attn_output = self.attention(pos_emb)

        _, (hy, _) = self.lstm(attn_output, (hn_repeat, cn_repeat))

        # Use only the last layer's output for prediction
        output = self.predict(hy[-1]).view(-1, self.enc_in, self.pred_len)
        return output
    # def parallel_multi_step_forecasting_decode(self, hn, cn, batch_size):
    #     pos_emb = self.position_embed_get(batch_size)
        
    #     # Adjust for multiple layers
    #     num_layers = self.lstm.num_layers
        
    #     # Reshape hn and cn to account for multiple layers
    #     hn_repeat = hn.repeat(1, 1, self.seg_num_y).view(num_layers, -1, self.d_model)
    #     cn_repeat = cn.repeat(1, 1, self.seg_num_y).view(num_layers, -1, self.d_model)

    #     # Directly use pos_emb as input to LSTM without attention
    #     _, (hy, _) = self.lstm(pos_emb, (hn_repeat, cn_repeat))

    #     # Use only the last layer's output for prediction
    #     output = self.predict(hy[-1]).view(-1, self.enc_in, self.pred_len)
    #     return output

    def position_embed_get(self, batch_size):
        if self.channel_id:
            pos_emb = torch.cat([
                self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),
                self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1)
            ], dim=-1).view(-1, 1, self.d_model).repeat(batch_size, 1, 1)
        else:
            pos_emb = self.pos_emb.repeat(batch_size * self.enc_in, 1).unsqueeze(1)
        return pos_emb

    def permute_and_denormalize(self, y):
        y = y.permute(0, 2, 1)
        if self.revin:
            return self.revin_layer(y, 'denorm')
        else:
            return y + self.seq_last