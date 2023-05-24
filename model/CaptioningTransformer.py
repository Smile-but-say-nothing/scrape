import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import resnet50, googlenet, resnet34, inception_v3, vgg


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
        # attention: (B, n_head, L, L), v: (B, n_head, L, d_k
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
    
    def forward(self, dec_input, enc_output, pad_subsequent_mask=None):
        # dec_input: (B, L, d_model)
        # slf_attn_mask: (B, L, L) i.e. pad_mask + sub_attention_mask, dec_enc_attn_mask: (B, 1, L) i.e. pad_mask
        # https://blog.csdn.net/weixin_42253689/article/details/113838263
        # for masked multi-head attention, we use slf_attn_mask (pad_mask + sub_attention_mask)
        output, masked_self_attention = self.masked_multi_head_attention(dec_input, dec_input, dec_input, mask=pad_subsequent_mask)
        # for second multi-head attention, we use pad_mask only
        output, self_attention = self.multi_head_attention(output, enc_output, enc_output, mask=None)
        # dec_output: (B, L, d_model)
        output = self.position_ffn(output)
        return output, masked_self_attention, self_attention


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, max_pos=100):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_pos_table(embedding_size, max_pos))
    
    def _get_pos_table(self, embedding_size, max_pos):
        def get_angles(pos, i, d_model):
            angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
            return pos * angle_rates  # broadcasting
        
        pos_col_vector = np.arange(max_pos)[:, np.newaxis]
        i_row_vector = np.arange(embedding_size)[np.newaxis, :]
        d_model = embedding_size
        pos_table = get_angles(pos_col_vector, i_row_vector, d_model)
        pos_table[:, 0::2] = np.sin(pos_table[:, 0::2])  # sin for indices 2i
        pos_table[:, 1::2] = np.cos(pos_table[:, 1::2])  # cos for indices 2i + 1
        # x: (B, L, embedding_size), pos_table: (1, L, embedding_size)
        # use broadcasting to add pos_table values on the corresponding embedding elements
        # clone和detach意味着着只做简单的数据复制，既不数据共享，也不对梯度共享，从此两个张量无关联。
        # print(torch.tensor(pos_table, dtype=x.dtype, device=x.device).unsqueeze(0).clone().detach().shape)
        return torch.FloatTensor(pos_table).unsqueeze(0)
    
    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


def positional_encoding_2d(x, row, col, d_model):
    # first d_model/2 encode row embedding and second d_model/2 encode column embedding
    row_pos = np.repeat(np.arange(row), col)[:, np.newaxis]
    col_pos = np.repeat(np.expand_dims(np.arange(col), 0), row, axis=0).reshape(-1, 1)
    
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates
    
    angle_rads_row = get_angles(row_pos, np.arange(d_model // 2)[np.newaxis, :], d_model // 2)
    angle_rads_col = get_angles(col_pos, np.arange(d_model // 2)[np.newaxis, :], d_model // 2)
    angle_rads_row[:, 0::2] = np.sin(angle_rads_row[:, 0::2])
    angle_rads_row[:, 1::2] = np.cos(angle_rads_row[:, 1::2])
    angle_rads_col[:, 0::2] = np.sin(angle_rads_col[:, 0::2])
    angle_rads_col[:, 1::2] = np.cos(angle_rads_col[:, 1::2])
    pos_encoding = np.concatenate([angle_rads_row, angle_rads_col], axis=1)[np.newaxis, ...]
    return x + torch.tensor(pos_encoding, dtype=x.dtype, device=x.device).clone().detach()


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, n_layer, n_head, d_k, d_v, d_model, d_ff, pad_idx, dropout, scale_emb):
        """
        A single encoder in Transformer
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_idx)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([EncoderLayer(d_model, d_ff, n_head, d_k, d_v, dropout) for _ in range(n_layer)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model
    
    def forward(self, src_seq, pad_mask=None, return_attention=False):
        self_attention_list = []
        output = src_seq
        # "In the embedding layers, we multiply those weights by \sqrt{d_model}."
        if self.scale_emb:
            output *= self.d_model ** 0.5
        
        output = positional_encoding_2d(output, 7, 7, self.d_model)
        
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        # output: (B, L, embedding_size)
        output = self.layer_norm(output)
        for enc_layer in self.layer_stack:
            output, self_attention = enc_layer(output, pad_mask=pad_mask)
            self_attention_list += [self_attention] if return_attention else []
        if return_attention:
            return output, self_attention_list
        return output,


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, n_layer, n_head, d_k, d_v, d_model, d_ff, pad_idx, dropout, scale_emb):
        """
            A single decoder in Transformer
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_idx)
        self.dropout = nn.Dropout(p=dropout)
        self.pe = PositionalEncoding(embedding_size)
        self.layer_stack = nn.ModuleList([DecoderLayer(d_model, d_ff, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layer)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model
    
    def forward(self, trg_seq, trg_mask, enc_output, return_attention=False):
        decoder_self_attention_list, decoder_encoder_self_attention_list = [], []
        # print(trg_seq)
        output = self.embedding(trg_seq)
        if self.scale_emb:
            output *= self.d_model ** 0.5
        # get seq_len and feed to PositionalEncoding
        output = self.dropout(self.pe(output))
        # dec_output: (B, L, d_model)
        output = self.layer_norm(output)
        
        for dec_layer in self.layer_stack:
            output, self_attention, encoder_self_attention = dec_layer(output, enc_output, pad_subsequent_mask=trg_mask)
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


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # model = inception_v3(pretrained=True, aux_logits=False)
        model = resnet34(pretrained=True)
        for param in model.parameters():
            param.requires_grad_(False)  # do not update encoder's gradient
        modules = list(model.children())[:-2]  # remove last 3 layers
        self.model = nn.Sequential(*modules)
    
    def forward(self, images):
        features = self.model(images)
        # features: (B, 8 * 8, 2048)
        features = features.view(features.size(0), features.size(1), -1).permute(0, 2, 1)
        return features


class CaptioningTransformer(nn.Module):
    def __init__(self, vocab_size, pad_idx=0, embedding_size=512, d_model=512, d_ff=2048,
                 n_layer=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
                 linear_and_embedding_weight_sharing=True,
                 scale_embedding=True):
        """
        TransformerForCaption
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
            linear_and_embedding_weight_sharing: weight sharing between linear transformation and 2 embedding layers
            scale_embedding: whether embedding divided by d_model ** 0.5
        Return:
            seq_logit
        """
        super().__init__()
        assert embedding_size == d_model, "embedding size should be equal to d_model for residual connection"
        self.pad_idx = pad_idx
        # image feature extractor
        self.extractor = FeatureExtractor()
        # image feature projection
        self.image_feature_projection = nn.Linear(2048, embedding_size)
        # encoder or decoder includes n_layer layers
        self.encoder = Encoder(vocab_size, embedding_size, n_layer, n_head, d_k, d_v, d_model, d_ff, pad_idx, dropout, scale_embedding)
        self.decoder = Decoder(vocab_size, embedding_size, n_layer, n_head, d_k, d_v, d_model, d_ff, pad_idx, dropout, scale_embedding)
        # linear
        self.linear = nn.Linear(d_model, vocab_size, bias=False)
        # relu
        self.relu = nn.ReLU()
        # init by xavier_uniform_
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        if linear_and_embedding_weight_sharing:
            self.linear.weight = self.decoder.embedding.weight
    
    def forward(self, features, captions, lengths):
        captions = captions[:, :-1]
        # Linear + ReLU
        features = self.image_feature_projection(features)
        # (6, 49, 512)
        features = self.relu(features)
        
        # get_pad_mask(trg_seq, self.pad_idx): (B, 1, L), get_subsequent_mask(trg_seq): (1, L, L)
        # use broadcasting to produce trg_mask: (B, L, L) which include pad_mask and subsequent_mask
        pad_subsequent_mask = get_pad_mask(captions, self.pad_idx) & get_subsequent_mask(captions)
        
        # forward
        dec_output, *_ = self.decoder(captions, pad_subsequent_mask, features)
        # dec_output: (B, L, d_model)
        seq_logit = self.linear(dec_output)
        # seq_logit: (B * L, vocab_size)
        seq_logit = seq_logit.view(-1, seq_logit.size(2))
        return seq_logit
    
    def sample(self, images, features, max_len=25, search_mode="beam", beam_size=10):
        output = torch.LongTensor(images.size(0), 1).fill_(1).to("cuda:0")
        features = self.image_feature_projection(features)
        features = self.relu(features)
        enc_output = features
        
        if search_mode == "greedy":
            for i in range(max_len):
                pad_subsequent_mask = get_pad_mask(output, self.pad_idx) & get_subsequent_mask(output)
                dec_output, *_ = self.decoder(output, pad_subsequent_mask, enc_output)
                seq_logit = self.linear(dec_output)
                predictions_id = torch.argmax(seq_logit[:, -1], dim=-1)
                output = torch.cat([output, predictions_id.unsqueeze(1)], dim=-1)
            return output.squeeze().data.cpu().numpy()  # squeeze dim 0
        else:
            feeds = torch.ones((1, 1), device="cuda:0", dtype=torch.long)  # initial feed, feeds: (1, 1)
            for i in range(max_len):
                if i == 0:
                    pad_subsequent_mask = get_pad_mask(feeds, self.pad_idx) & get_subsequent_mask(feeds)
                    dec_output, *_ = self.decoder(feeds, pad_subsequent_mask, enc_output)
                    seq_logit = self.linear(dec_output)
                    
                    scores = F.softmax(seq_logit.squeeze(), -1)  # scores: (vocab_size)
                    scores = scores.log()
                    scores = scores.unsqueeze(0)  # scores: (1, vocab_size)
                    sorted_scores, idxes = scores.topk(beam_size, dim=1)  # sorted_scores, idxes: (1, beam_size)

                    candidates = idxes.unsqueeze(-1)  # candidates: (1, beam_size, 1)
                    feeds = torch.reshape(candidates, (-1, 1))  # feeds: (10, 1)
                    V = scores.size(-1)  # V: vocab_size

                    enc_output = torch.cat([enc_output] * beam_size)
                else:
                    pad_subsequent_mask = get_pad_mask(feeds, self.pad_idx) & get_subsequent_mask(feeds)
                    dec_output, *_ = self.decoder(feeds, pad_subsequent_mask, enc_output)
                    seq_logit = self.linear(dec_output)
                    scores = F.softmax(seq_logit[:, -1].squeeze(), -1)  # scores: (vocab_size)
                    scores = scores.log() + sorted_scores.view(-1, 1)  # scores: (beam_size, vocab_size)
                    scores = torch.reshape(scores, (1, -1))  # flatten scores to get index by // and %, scores: (1, beam_size * vocab_size)
                    sorted_scores, idxes = scores.topk(beam_size, dim=1)  # sorted_scores, idxes: (1, 10)
                    prior_candidate = candidates[np.arange(1)[:, None], idxes // V]  # prior_candidate: (1, beam_size, t)
                    current_candidate = (idxes % V).unsqueeze(-1)  # current_candidate: (1, beam_size, 1)
                    candidates = torch.cat([prior_candidate, current_candidate], -1)  # candidates: (1, beam_size, t + 1)
                    feeds = candidates.squeeze()
            captions = candidates[:, 0, :].squeeze().data.cpu().numpy()  # pick best candidates
            return captions


            
        




