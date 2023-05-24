from backbone import resnet, ViT
import json
from utils.clean_captions import clean_caption
import logging
import torch.nn as nn
from torchvision.models import resnet101
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
import wandb
from model import EncoderCNN, DecoderRNN, NIC, CaptioningTransformer, SoftAttention
from torchvision import transforms

def build_wandb(name, config):
    wandb_config = dict(
        dataset_path=config.dataset_path,
        cnn_learning_rate=config.extractor_lr,
        caption_learning_rate=config.caption_lr,
        batch_size=config.batch_size,
        end_epoch=config.end_epoch,
        weight_decay=config.weight_decay,
        captions_per_image=config.captions_per_image
    )
    experiment = wandb.init(
        project='image caption',
        entity='scrape',
        name=name,
        id=config.wand_id,
        allow_val_change=True,
        resume=config.resume_last,
        config=wandb_config,
    )
    return experiment
    
def build_data_transform(split):
    data_transform = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    }
    return data_transform[split]

def build_feature_extractor(feature_extractor, pretrained=True, finetune=True):
    if feature_extractor in ['resnet101']:
        # Pytorch pretrained ImageNet ResNet-101
        # cnn_model = resnet101(pretrained=pretrained)
        # Our pretrained ImageNet ResNet-101
        cnn_model = getattr(resnet, feature_extractor)(pretrained=pretrained)
        for p in cnn_model.parameters():
            p.requires_grad = False
        # only fine-tune convolutional blocks 2 through 4
        for c in list(cnn_model.children())[5:]:
            for p in c.parameters():
                p.requires_grad = finetune
        # cnn_model = list(cnn_model.children())[:-3]
        # cnn_model = nn.Sequential(*cnn_model)  # (6, 1024, 14, 14)
        return cnn_model
    elif feature_extractor in ['ViT']:
        vit = ViT('vit_base_patch32_224', pretrained=pretrained)
        for p in vit.parameters():
            p.requires_grad = finetune
        for p in vit.fc.parameters():
            # must be True
            p.requires_grad = True
        return vit
        
def build_caption_model(vocab_size, caption_model, model_setting, config):
    if caption_model == 'NIC':
        caption_model = NIC(
            embed_size=model_setting.embed_size,
            hidden_size=model_setting.hidden_size,
            vocab_size=vocab_size,
            num_layers=model_setting.num_layers,
            unit=model_setting.unit).to(config.device)
    elif caption_model == 'SoftAttention':
        caption_model = SoftAttention(
            embed_size=model_setting.embed_size,
            hidden_size=model_setting.hidden_size,
            vocab_size=vocab_size
        ).to(config.device)
    elif caption_model == 'Transformer':
        caption_model = CaptioningTransformer(
            vocab_size,
            pad_idx=model_setting.pad_idx,
            embedding_size=model_setting.embedding_size,
            d_model=model_setting.d_model,
            d_ff=model_setting.d_ff,
            n_layer=model_setting.n_layer,
            n_head=model_setting.n_head,
            d_k=model_setting.d_k,
            d_v=model_setting.d_v,
            dropout=model_setting.dropout,
            linear_and_embedding_weight_sharing=model_setting.linear_and_embedding_weight_sharing,
            scale_embedding=model_setting.scale_embedding).to(config.device)
    return caption_model

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold
    
    def __len__(self):
        return len(self.itos)
    
    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4
        for word in sentence_list:
            if word not in frequencies:
                frequencies[word] = 1
            else:
                frequencies[word] += 1
            
            if frequencies[word] == self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
    
    def numericalize(self, text):
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in text
        ]


def build_vocab(json_root, clean=False, freq_threshold=1):
    with open(json_root, 'r', encoding='gbk') as f:
        data = json.load(f)
    raw_caption = []
    for e in data['annotations']:
        for s in e['sentences']:
            if clean:
                raw_caption.extend(clean_caption(s['tokens']))
            else:
                raw_caption.extend(s['tokens'])
    
    vocab = Vocabulary(freq_threshold)
    vocab.build_vocabulary(raw_caption)
    return vocab


class Scorer():
    def __init__(self, hypothesis, references):
        self.hypothesis = hypothesis
        self.references = references
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            # (Spice(), "SPICE"),
        ]
    
    def compute_scores(self):
        total_scores = {}
        for scorer, method in self.scorers:
            score, scores = scorer.compute_score(self.references, self.hypothesis)
            if type(method) == list:
                # for sc, scs, m in zip(score, scores, method):
                #     print("%s: %0.3f" % (m, sc))
                total_scores["Bleu"] = score
            else:
                # print("%s: %0.3f" % (method, score))
                total_scores[method] = score
        return total_scores

def build_scorer(hypothesis, references):
    scorer = Scorer(hypothesis, references)
    return scorer
