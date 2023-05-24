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


class DecoderRNN(nn.Module):
    __units = {"elman": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
    
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, unit='gru'):
        """
        Args:
            embed_size: embedding size of each word
            hidden_size: size of hidden weight in RNN
            vocab_size: size of vocab
            num_layers: layers of RNN
            unit: type of RNN used
        """
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.unit_name = unit
        # Model
        self.unit = DecoderRNN.__units[unit](embed_size, hidden_size, num_layers, batch_first=True)
        # word embedding
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=embed_size, padding_idx=0)
        self.relu = nn.ReLU(inplace=True)
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)
        
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
        # image_caption_embed: (B, L + 1, Embed_size)
        image_caption_embed = torch.cat((features, caption_embed), dim=1)
        # lengths: list of sequences lengths of each batch element
        inputs_packed = pack_padded_sequence(image_caption_embed, lengths, batch_first=True)
        outputs, _ = self.unit(inputs_packed)  # outputs[0]: (*, Hidden_size)
        outputs = self.relu(self.linear(outputs[0]))  # outputs[0]: (*, Vocab_size)
        return outputs
    
    def sample(self, features, max_len=25, search_mode="beam", beam_size=10):
        """
        Sample from RNN using greedy search or beam search
        Args:
            features: features from CNN feature extractor
            max_len: max length of sentences
            search_mode: sample mode, greedy search or beam search
            beam_size: beam search size
        Return:
            list of predicted image captions
        """
        output_ids = []
        states = None
        inputs = features
        if search_mode == "greedy":
            for i in range(max_len):
                # pass data through recurrent network
                outputs, states = self.unit(inputs, states)
                outputs = self.relu(self.linear(outputs.squeeze(1)))  # outputs: (1, vocab_size)
                # find maximal predictions
                predicted = outputs.max(1)[1]  # get maximal indices i.e. max()[1], specific [*]
                # append results from given step to global results
                output_ids.append(predicted)
                # prepare chosen words for next decoding step
                inputs = self.embedding(predicted)
                inputs = inputs.unsqueeze(1)  # inputs:(1, 1, embed_size)
            output_ids = torch.stack(output_ids, 1)  # output_ids: (1, max_len)
            return output_ids.squeeze().data.cpu().numpy()  # squeeze dim 0
        if search_mode == "beam":
            feeds = torch.ones((1, 1), device="cuda:0", dtype=torch.long)  # initial feed, feeds: (1, 1)
            hidden = None
            for t in range(max_len):
                if t == 0:  # t == 0 is specified for image input
                    outputs, hidden = self.unit(inputs)
                    outputs = self.relu(self.linear(outputs.squeeze(1)))
                    
                    scores = F.softmax(outputs.squeeze(), -1)  # scores: (vocab_size)
                    scores = scores.log()
                    scores = scores.unsqueeze(0)  # scores: (1, vocab_size)
                    sorted_scores, idxes = scores.topk(beam_size, dim=1)  # sorted_scores, idxes: (1, beam_size)
 
                    candidates = idxes.unsqueeze(-1)  # candidates: (1, beam_size, 1)
                    feeds = torch.reshape(candidates, (-1, 1))  # feeds: (10, 1)

                    V = scores.size(-1)  # V: vocab_size
                    if self.unit_name == "gru":
                        # hidden: (1, beam_size, hidden_size)
                        hidden = torch.reshape(torch.cat([hidden.squeeze(0)] * beam_size, -1), (hidden.size(0), -1, hidden.size(-1)))
                    else:
                        # LSTM has h0 and c0
                        h, c = hidden
                        h = torch.reshape(torch.cat([h] * beam_size, -1), (h.size(0), -1, h.size(-1)))  # (L, N*B, H)
                        c = torch.reshape(torch.cat([c] * beam_size, -1), (c.size(0), -1, c.size(-1)))
                        hidden = (h, c)
                else:
                    # feeds: (10, 1) or (beam_size, 1) means 10 sentences which each one includes one word
                    caption_embed = self.embedding(feeds)  # caption_embed: (beam_size, 1, embed_size)
                    scores, hidden = self.unit(caption_embed, hidden)  # scores: (beam_size, 1, embed_size)
                    scores = self.relu(self.linear(scores))
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


class NIC(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, unit='gru'):
        super(NIC, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers, unit)
        
    def forward(self, features, captions, lengths):
        features = self.encoder(features)
        outputs = self.decoder(features, captions, lengths)
        return outputs
        
        
        