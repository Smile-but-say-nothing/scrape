import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        self.embed = nn.Linear(2048, embed_size)
    
    def forward(self, features):
        features = features.mean(1).unsqueeze(1)
        features = self.embed(features)
        return features

class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha

class DecoderRNN(nn.Module):
    __units = {"elman": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
    
    def __init__(self, embed_size, hidden_size, vocab_size):
        """
        Args:
            embed_size: embedding size of each word
            hidden_size: size of hidden weight in RNN
            vocab_size: size of vocab
        """
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        # word embedding
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=embed_size, padding_idx=0)
        self.relu = nn.ReLU(inplace=True)
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)

        self.attention = Attention(embed_size, embed_size, hidden_size)  # attention network
        self.decode_step = nn.LSTMCell(embed_size + embed_size, embed_size, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(embed_size, embed_size)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(embed_size, embed_size)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(embed_size, embed_size)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(embed_size, vocab_size)  # linear layer to find scores over vocabulary

    
    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c
    
    def forward(self, features, captions, lengths):
        """
        Shapes:
            features: (B, Embed_size)
            captions: (B, L)
            pattern: Connect Method
        Return: (B, L, Vocab_size)
        """
        # features: (B, Embed_size) -> (B, 1, Num_features), caption_embed: (B, L, Embed_size)
        caption_embed = self.embedding(captions)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = torch.sort(torch.tensor(lengths, dtype=torch.int), dim=0, descending=True)
        encoder_out = features[sort_ind]
        embeddings = caption_embed[sort_ind]

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)  
        
        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(features.size(0), max(lengths), self.vocab_size).to('cuda:0')
        alphas = torch.zeros(features.size(0), max(lengths), features.size(1)).to('cuda:0')

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(lengths)):
            batch_size_t = sum([l > t for l in lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(h)  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha
        # print(predictions.shape)
        # print(pack_padded_sequence(predictions, lengths, batch_first=True), len(pack_padded_sequence(predictions, lengths, batch_first=True)[1]))
        predictions = pack_padded_sequence(predictions, lengths, batch_first=True)[0]
        
        return predictions

    
    def sample(self, features, max_len=25, search_mode="beam", beam_size=10):
        feeds = torch.ones((1, 1), device="cuda:0", dtype=torch.long)  # initial feed, feeds: (1, 1)
        h, c = self.init_hidden_state(features)
        inputs = self.embedding(feeds).squeeze(1)  # (s, embed_dim)
        output_ids = []
        if search_mode == "greedy":
            for i in range(max_len):
                awe, _ = self.attention(features, h)  # (s, encoder_dim), (s, num_pixels)
                gate = self.sigmoid(self.f_beta(h))  # gating scalar, (s, encoder_dim)
                awe = gate * awe
                print(inputs.shape, awe.shape)
                h, c = self.decode_step(torch.cat([inputs, awe], dim=1), (h, c))  # (s, decoder_dim)
                outputs = self.fc(h)  # (s, vocab_size)
                predicted = outputs.max(1)[1]  # get maximal indices i.e. max()[1], specific [*]
                # append results from given step to global results
                output_ids.append(predicted)
                # prepare chosen words for next decoding step
                inputs = self.embedding(predicted)
            output_ids = torch.stack(output_ids, 1)  # output_ids: (1, max_len)
            return output_ids.squeeze().data.cpu().numpy()  # squeeze dim 0
        else:
            for t in range(max_len):
                if t == 0:  # t == 0 is specified for image input
                    awe, _ = self.attention(features, h)  # (s, encoder_dim), (s, num_pixels)
                    gate = self.sigmoid(self.f_beta(h))  # gating scalar, (s, encoder_dim)
                    awe = gate * awe
                    
                    embeddings = self.embedding(feeds).squeeze(1)  # (s, embed_dim)
                    h, c = self.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)
                    
                    scores = self.fc(h)  # (s, vocab_size)
                    scores = F.softmax(scores.squeeze(), -1)  # scores: (vocab_size)
                    scores = scores.log()
                    scores = scores.unsqueeze(0)  # scores: (1, vocab_size)
                    sorted_scores, idxes = scores.topk(beam_size, dim=1)  # sorted_scores, idxes: (1, beam_size)
    
                    candidates = idxes.unsqueeze(-1)  # candidates: (1, beam_size, 1)
                    feeds = torch.reshape(candidates, (-1, 1))  # feeds: (10, 1)
                    V = scores.size(-1)  # V: vocab_size
    
                    h = torch.reshape(torch.cat([h] * beam_size, -1), (-1, h.size(-1)))
                    c = torch.reshape(torch.cat([c] * beam_size, -1), (-1, c.size(-1)))
                    
                else:
                    # feeds: (10, 1) or (beam_size, 1) means 10 sentences which each one includes one word
                    embeddings = self.embedding(feeds).squeeze(1)  # caption_embed: (beam_size, 1, embed_size)
                    awe, alpha = self.attention(features, h)
                    gate = self.sigmoid(self.f_beta(h))  # gating scalar, (s, encoder_dim)
                    awe = gate * awe
                    h, c = self.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)
                    scores = self.fc(h)  # (s, vocab_size)
                    
                    scores = F.softmax(scores.squeeze(), -1)  # scores: (beam_size, vocab_size)
                    # scores: (beam_size, vocab_size), sorted_scores: (beam_size, 1) and broadcasting
                    scores = scores.log() + sorted_scores.view(-1, 1)  # scores: (beam_size, vocab_size)
                    scores = torch.reshape(scores, (1, -1))  # flatten scores to get index by // and %, scores: (1, beam_size * vocab_size)
                    sorted_scores, idxes = scores.topk(beam_size, dim=1)  # sorted_scores, idxes: (1, 10)
                    # Ref: https://cloud.tencent.com/developer/article/1675791
                    prior_candidate = candidates[np.arange(1)[:, None], idxes // V]  # prior_candidate: (1, beam_size, t)
                    current_candidate = (idxes % V).unsqueeze(-1)  # current_candidate: (1, beam_size, 1)
                    candidates = torch.cat([prior_candidate, current_candidate], -1)  # candidates: (1, beam_size, t + 1)
                    feeds = torch.reshape(candidates[:, :, -1], (-1, 1))  # feeds are for the next loop. feeds: (beam_size, 1)
    
            captions = candidates[:, 0, :].squeeze().data.cpu().numpy()  # pick best candidates
        return captions

class SoftAttention(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(SoftAttention, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
    
    def forward(self, features, captions, lengths):
        features = self.encoder(features)
        outputs = self.decoder(features, captions, lengths)
        return outputs


