import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import resnet50, googlenet, resnet34

class ScaledDotProductAttention(nn.Module):
    def __init__(self, scale):
        """
        ScaledDotProductAttention Mechanism
        Args:
            scale: d_k ** 0.5
        Return:
            output: (B, n_head, L, d_k)
            attention: (B, n_head, L, L)
        """
        super(ScaledDotProductAttention, self).__init__()
        self.scale = scale
    
    def forward(self, q, k, v, mask=None):
        # q, k, v: (B, n_head, L, d_k), attention: (B, n_head, L, L)
        attention = torch.matmul(q, k.transpose(2, 3)) / self.scale
        if mask is not None:
            # mask: (B, 1, 1, L)
            attention = attention.masked_fill(mask == 0, -1e9)  # -1e9 where mask == 0 i.e. <PAD>
        attention = F.softmax(attention, dim=-1)
        # attention: (B, n_head, L, L), v: (B, n_head, L, d_k)
        output = torch.matmul(attention, v)
        return output, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout):
        """
        MultiHeadAttention Mechanism
        Args:
            n_head: number of heads
            d_model: embedding size
            d_k: dimension of key vector
            d_v: dimension of value vector
            dropout: dropout prob
        """
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.linear = nn.Linear(n_head * d_v, d_model, bias=False)
        
        self.attention = ScaledDotProductAttention(scale=d_k ** 0.5)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    
    def forward(self, q, k, v, mask=None):
        # q, k, v: (B, L, d_model)
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        
        # residual: (B, L, d_model)
        residual = q
        # (B, L, d_model) -> (B, L, n_head * d_k) -> (B, L, n_head, d_k)
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        
        # Transpose for q, k, v: (B, L, n_head, d_k) -> (B, n_head, L, d_k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        if mask is not None:
            # mask: (B, 1, 1, L), mask is aimed to avoid the attention calculation between <PAD> and words
            mask = mask.unsqueeze(1)
        # ScaledDotProductAttention
        q, attention = self.attention(q, k, v, mask=mask)
        
        # q: (B, n_head, L, d_k) -> (B, L, n_head * d_k), attn: (B, n_head, L, L)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)  # contiguous + view
        # q: (B, L, n_head * d_k) -> (B, L, d_model)
        q = self.dropout(self.linear(q))
        q += residual
        q = self.layer_norm(q)
        
        return q, attention


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        """
        PositionWiseFeedForward
        Args:
            d_model: output dimension of all sub-layers
            d_ff: inner hidden size of feed-forward networks
            dropout: dropout prob
        """
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (B, L, d_model)
        residual = x
        
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        
        x = self.layer_norm(x)
        
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_head, d_k, d_v, dropout):
        """
        EncoderLayer = MultiHeadAttention (include add & norm) + PositionWiseFeedForward (include add & norm)
        """
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.position_ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
    
    def forward(self, enc_input, pad_mask=None):
        output, self_attention = self.multi_head_attention(enc_input, enc_input, enc_input, mask=pad_mask)
        output = self.position_ffn(output)
        return output, self_attention


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_head, d_k, d_v, dropout):
        """
        DecoderLayer = Masked MultiHeadAttention (include add & norm) + MultiHeadAttention (include add & norm) + PositionWiseFeedForward (include add & norm)
        """
        super(DecoderLayer, self).__init__()
        self.masked_multi_head_attention = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.multi_head_attention = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.position_ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
    
    def forward(self, dec_input, enc_output, pad_subsequent_mask=None, pad_mask=None):
        # dec_input: (B, L, d_model)
        # slf_attn_mask: (B, L, L) i.e. pad_mask + sub_attention_mask, dec_enc_attn_mask: (B, 1, L) i.e. pad_mask
        # https://blog.csdn.net/weixin_42253689/article/details/113838263
        # for masked multi-head attention, we use slf_attn_mask (pad_mask + sub_attention_mask)
        output, masked_self_attention = self.masked_multi_head_attention(dec_input, dec_input, dec_input, mask=pad_subsequent_mask)
        # for second multi-head attention, we use pad_mask only
        output, self_attention = self.multi_head_attention(output, enc_output, enc_output, mask=pad_mask)
        # dec_output: (B, L, d_model)
        output = self.position_ffn(output)
        return output, masked_self_attention, self_attention


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        
        # Not a parameter
        # pos_table: (1, n_position, embedding_size)
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))
    
    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        
        # sinusoid_table: (n_position, embedding_size)
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        # step = 2
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)
    
    def forward(self, x):
        # x: (B, L, embedding_size)
        # use broadcasting to add pos_table values on the corresponding embedding elements
        # clone和detach意味着着只做简单的数据复制，既不数据共享，也不对梯度共享，从此两个张量无关联。
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, n_layer, n_head, d_k, d_v, d_model, d_ff, pad_idx, dropout, n_position, scale_emb):
        """
        A single encoder in Transformer
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(embedding_size, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([EncoderLayer(d_model, d_ff, n_head, d_k, d_v, dropout) for _ in range(n_layer)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model
    
    def forward(self, src_seq, pad_mask, return_attention=False):
        self_attention_list = []
        # output: (B, L, embedding_size)
        output = self.embedding(src_seq)
        # "In the embedding layers, we multiply those weights by \sqrt{d_model}."
        if self.scale_emb:
            output *= self.d_model ** 0.5
        output = self.dropout(self.positional_encoding(output))
        # output: (B, L, embedding_size)
        output = self.layer_norm(output)
        for enc_layer in self.layer_stack:
            output, self_attention = enc_layer(output, pad_mask=pad_mask)
            self_attention_list += [self_attention] if return_attention else []
        if return_attention:
            return output, self_attention_list
        return output,


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, n_layer, n_head, d_k, d_v, d_model, d_ff, pad_idx, dropout, n_position, scale_emb):
        """
            A single decoder in Transformer
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(embedding_size, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([DecoderLayer(d_model, d_ff, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layer)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model
    
    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attention=False):
        decoder_self_attention_list, decoder_encoder_self_attention_list = [], []
        output = self.embedding(trg_seq)
        if self.scale_emb:
            output *= self.d_model ** 0.5
        output = self.dropout(self.positional_encoding(output))
        # dec_output: (B, L, d_model)
        output = self.layer_norm(output)
        
        for dec_layer in self.layer_stack:
            output, self_attention, encoder_self_attention = dec_layer(output, enc_output, pad_subsequent_mask=trg_mask, pad_mask=src_mask)
            decoder_self_attention_list += [self_attention] if return_attention else []
            decoder_encoder_self_attention_list += [encoder_self_attention] if return_attention else []
        if return_attention:
            return output, decoder_self_attention_list, decoder_encoder_self_attention_list
        return output,


def get_pad_mask(seq, pad_idx):
    # get pad mask from seq, return shape: (B, 1, L)
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    batch_size, seq_len = seq.size()
    # torch.triu returns the upper triangular part of a matrix, others are set 0
    # subsequent_mask: (1, L, L)
    subsequent_mask = (1 - torch.triu(torch.ones((1, seq_len, seq_len), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class Transformer(nn.Module):
    def __init__(self, vocab_size, pad_idx=0, embedding_size=512, d_model=512, d_ff=2048,
                 n_layer=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
                 linear_and_embedding_weight_sharing=True, src_and_trg_embedding_weight_sharing=True,
                 scale_embedding=True):
        """
        Transformer
        Args:
            vocab_size: vocab size of union data (source + target)
            pad_idx: idx of padding
            embedding_size: size of embedding layers, equals to d_model
            d_model: output dimension of all sub-layers
            d_ff: inner hidden size of feed-forward networks
            n_layer: layer numbers of encoders and decoders
            n_head: head numbers of multi-head attention
            d_k: dimension of vector key, equals to d_model / n_head
            d_v: dimension of vector value, equals to d_model / n_head
            dropout: dropout prob after each sub-layer, before the sum of embedding and PE
            n_position: the length of generated Positional Encoding
            linear_and_embedding_weight_sharing: weight sharing between linear transformation and 2 embedding layers
        Return:
            seq_logit
        """
        super().__init__()
        assert embedding_size == d_model, "embedding size should be equal to d_model for residual connection"
        self.pad_idx = pad_idx
        # encoder or decoder includes n_layer layers
        self.encoder = Encoder(vocab_size, embedding_size, n_layer, n_head, d_k, d_v, d_model, d_ff, pad_idx, dropout, n_position, scale_embedding)
        self.decoder = Decoder(vocab_size, embedding_size, n_layer, n_head, d_k, d_v, d_model, d_ff, pad_idx, dropout, n_position, scale_embedding)
        # linear
        self.linear = nn.Linear(d_model, vocab_size, bias=False)
        # init by xavier_uniform_
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # shared weight between 2 embedding layers and pre-softmax linear transformation
        if src_and_trg_embedding_weight_sharing:
            self.decoder.embedding.weight = self.encoder.embedding.weight
        if linear_and_embedding_weight_sharing:
            self.linear.weight = self.decoder.embedding.weight
    
    def forward(self, src_seq, trg_seq):
        # src_mask: (B, 1, L)
        pad_mask = get_pad_mask(src_seq, self.pad_idx)
        # get_pad_mask(trg_seq, self.pad_idx): (B, 1, L), get_subsequent_mask(trg_seq): (1, L, L)
        # use broadcasting to produce trg_mask: (B, L, L) which include pad_mask and subsequent_mask
        pad_subsequent_mask = get_pad_mask(trg_seq, self.pad_idx) & get_subsequent_mask(trg_seq)
        
        # forward
        enc_output, *_ = self.encoder(src_seq, pad_mask)
        dec_output, *_ = self.decoder(trg_seq, pad_subsequent_mask, enc_output, pad_mask)
        # dec_output: (B, L, d_model)
        seq_logit = self.linear(dec_output)
        # seq_logit: (B * L, vocab_size)
        seq_logit = seq_logit.view(-1, seq_logit.size(2))
        print(seq_logit.shape)
        return seq_logit