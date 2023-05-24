import torch
import os
from my_dataset import ShaCaption
from torch.utils.data import DataLoader
from utils import build_feature_extractor, build_vocab, build_caption_model, build_wandb, build_data_transform, clean_sos_eos, set_seed
from utils.build_base import build_scorer
from utils.collate_fn import MyCollate
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
import random
import wandb
from omegaconf.dictconfig import DictConfig
import logging
from itertools import groupby
from omegaconf import OmegaConf
import torch.nn.functional as F

os.environ['WANDB_MODE'] = 'offline'


def train_xe(feature_extractor, caption_model, dataloader, cnn_optimizer, optimizer, criterion, multilabel_criterion, config):
    feature_extractor.train()
    caption_model.train()
    step_loss, epoch_loss = 0.0, 0.0
    for step, (images, captions, lengths, multilabel) in enumerate(dataloader):
        images, captions, multilabel = images.to(config.device), captions.to(config.device), multilabel.to(config.device)
        decode_lengths = [l - 1 for l in lengths]
        optimizer.zero_grad()
        cnn_optimizer.zero_grad()
        # att_feats: (6, 49 or 50, 2048), fc_feats:(6, 9)
        att_feats, fc_feats = feature_extractor(images)
        outputs = caption_model(att_feats, captions, decode_lengths)
        # compute loss
        if getattr(config, 'caption_model') in ['NIC', 'SoftAttention']:
            targets = pack_padded_sequence(captions[:, 1:], decode_lengths, batch_first=True)[0]
        elif getattr(config, 'caption_model') in ['Transformer']:
            targets = captions[:, 1:].contiguous().view(-1)
        # outputs: (*, Vocab_size), targets: (*)
        if config['multi_task']:
            loss1 = criterion(outputs, targets)
            loss2 = multilabel_criterion(fc_feats, multilabel)
            loss = 0.8 * loss1 + 0.2 * loss2
            logging.info(f"Step [{step + 1}/{len(dataloader)}], Train Loss (0.8 * C + 0.2 * M): {loss.item():.3f},"
                         f"'Caption_loss': {loss1.item():.3f}, 'Multilabel_loss': {loss2.item():.3f}")
        else:
            loss = criterion(outputs, targets)
            logging.info(f"Step [{step + 1}/{len(dataloader)}], "
                         f"Train Loss (1.0 * C): {loss.item():.3f}")
        step_loss += loss.item()
        loss.backward()
        optimizer.step()
        cnn_optimizer.step()
    epoch_loss = step_loss / len(dataloader)
    return epoch_loss


def train_scst():
    return 1


def evaluate_loss(feature_extractor, caption_model, dataloader, criterion, multilabel_criterion, config):
    feature_extractor.eval()
    caption_model.eval()
    step_loss, epoch_loss = 0.0, 0.0
    with torch.no_grad():
        for step, (images, captions, lengths, multilabel) in enumerate(dataloader):
            images, captions, multilabel = images.to(config.device), captions.to(config.device), multilabel.to(config.device)
            decode_lengths = [l - 1 for l in lengths]
            if getattr(config, 'caption_model') in ['NIC', 'SoftAttention']:
                targets = pack_padded_sequence(captions[:, 1:], decode_lengths, batch_first=True)[0]
            elif getattr(config, 'caption_model') in ['Transformer']:
                targets = captions[:, 1:].contiguous().view(-1)
            # get features
            att_feats, fc_feats = feature_extractor(images)
            outputs = caption_model(att_feats, captions, decode_lengths)
            if config['multi_task']:
                loss1 = criterion(outputs, targets)
                loss2 = multilabel_criterion(fc_feats, multilabel)
                loss = 0.8 * loss1 + 0.2 * loss2
                logging.info(f"Step [{step + 1}/{len(dataloader)}], Val Loss (0.8 * C + 0.2 * M): {loss.item():.3f},"
                             f"'Caption_loss': {loss1.item():.3f}, 'Multilabel_loss': {loss2.item():.3f}")
            else:
                loss = criterion(outputs, targets)
                logging.info(f"Step [{step + 1}/{len(dataloader)}], "
                             f"Val Loss (1.0 * C): {loss.item():.3f}")
            step_loss += loss.item()
        epoch_loss = step_loss / len(dataloader)
        return epoch_loss


def evaluate_metrics(feature_extractor, caption_model, dataloader, vocab, config):
    feature_extractor.eval()
    caption_model.eval()
    hypothesis, references = {}, {}
    for step, (images, captions, lengths, multilabel) in enumerate(dataloader):
        images, captions, multilabel = images.to(config.device), captions.to(config.device), multilabel.to(config.device)
        att_feats, fc_feats = feature_extractor(images)
        # encode and decode
        sampled_ids = []
        if getattr(config, 'caption_model') in ['NIC', 'SoftAttention']:
            att_feats = caption_model.encoder(att_feats)
            sampled_ids = caption_model.decoder.sample(att_feats, max_len=50, search_mode="beam", beam_size=10)
        elif getattr(config, 'caption_model') in ['Transformer']:
            sampled_ids = caption_model.sample(images, att_feats, max_len=50, search_mode='beam', beam_size=10)
        if step % config.captions_per_image == 0:
            h = clean_sos_eos([vocab.itos[x] for x in sampled_ids])
            if config.group:
                h = [k for k, v in groupby(h)]
            h = [' '.join(h)]
            r = [' '.join(clean_sos_eos([vocab.itos[x] for x in captions.cpu().data.numpy()[0]]))]
            hypothesis.update({str(step): h})
            references.update({str(step): r})
        if step % config.captions_per_image != 0:
            r = ' '.join(clean_sos_eos([vocab.itos[x] for x in captions.cpu().data.numpy()[0]]))
            references[str(step - step % config.captions_per_image)].append(r)
        if step == len(dataloader) - 1:
            last_logit = F.softmax(fc_feats, dim=1).squeeze()
            last_multilabel = multilabel.squeeze().tolist()
    assert len(hypothesis) == len(references)
    logging.info(f"Caption of one case - Sample: {hypothesis['0']}, "
                 f"Target: {references['0']}")
    if config['multi_task']:
        itol = {0: '平和质', 1: '气虚质', 2: '阳虚质', 3: '阴虚质', 4: '痰湿质',
                5: '湿热质', 6: '血瘀质', 7: '气郁质', 8: '特禀质'}
        last_logit_sort = torch.sort(last_logit, descending=True)
        last_logit_sort_0, last_logit_sort_1 = last_logit_sort[0].tolist(), last_logit_sort[1].tolist()
        label_print = {str(itol[last_logit_sort_1[i]]): '%.3f' % (last_logit_sort_0[i]) for i in range(len(last_logit))}
        logging.info(f"Label of one case - Sample: {label_print}, "
                     f"Target: {' '.join([itol[idx] for idx, l in enumerate(last_multilabel) if l == 1])}")
    
    scorer = build_scorer(hypothesis, references)
    score = scorer.compute_scores()
    return score


def main(config: DictConfig):
    feature_extractor = getattr(config, 'feature_extractor')
    caption_model = getattr(config, 'caption_model')
    if config['multi_task']:
        wandb_name = f"{feature_extractor}-{caption_model}-{str(config[caption_model])}"
    else:
        wandb_name = f"{feature_extractor}-{caption_model}-{str(config[caption_model])}-w/o MTL"
    logging.info(wandb_name)
    experiment = build_wandb(name=wandb_name, config=config)
    wandb.config.update(config[caption_model])
    set_seed(config.seed)
    train_transform = build_data_transform('train')
    val_transform = build_data_transform('train')
    vocab = build_vocab(config.json_root, clean=config.clean, freq_threshold=config.freq_threshold)
    train_dataset = ShaCaption(config.json_root, train_transform, split='Train', segmented=config.segmented,
                               clean=config.clean, captions_per_image=config.captions_per_image, vocab=vocab)
    val_dataset = ShaCaption(config.json_root, val_transform, split='Val', segmented=config.segmented,
                             clean=config.clean, captions_per_image=config.captions_per_image, vocab=vocab)
    test_dataset = ShaCaption(config.json_root, val_transform, split='Test', segmented=config.segmented,
                              clean=config.clean, captions_per_image=config.captions_per_image, vocab=vocab)
    
    logging.info(f"image shape: {train_dataset[0][0].shape}, caption shape: {train_dataset[0][1].shape}, multilabel shape: {train_dataset[0][2].shape}")
    
    pad_idx = vocab.stoi["<PAD>"]
    loader_args = dict(batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True)
    train_data_loader = DataLoader(train_dataset, shuffle=True, collate_fn=MyCollate(pad_idx=pad_idx), **loader_args)
    val_data_loader = DataLoader(val_dataset, shuffle=False, collate_fn=MyCollate(pad_idx=pad_idx), **loader_args)
    
    # For sample
    loader_args['batch_size'] = 1
    val_data_loader_bs_1 = DataLoader(val_dataset, shuffle=False, collate_fn=MyCollate(pad_idx=pad_idx), **loader_args)
    test_data_loader_bs_1 = DataLoader(test_dataset, shuffle=False, collate_fn=MyCollate(pad_idx=pad_idx), **loader_args)
    
    vocab_size = len(vocab)
    logging.info(f"vocab_size:{vocab_size}")
    
    feature_extractor = build_feature_extractor(config.feature_extractor, pretrained=config.pretrained, finetune=config.finetune)
    feature_extractor.to(config.device)
    
    model_setting = getattr(config, caption_model)
    caption_model = build_caption_model(vocab_size, caption_model, model_setting, config)
    
    feature_extractor_param_num = sum(p.numel() for p in feature_extractor.parameters() if p.requires_grad)
    caption_model_param_num = sum(p.numel() for p in caption_model.parameters() if p.requires_grad)
    logging.info(f"feature_extractor - {feature_extractor_param_num / 1e6 :.2f}M, caption_model - {caption_model_param_num / 1e6 :.2f}M")
    
    cnn_optimizer = torch.optim.Adam([p for p in feature_extractor.parameters() if p.requires_grad], lr=config.extractor_lr)
    optimizer = torch.optim.Adam(caption_model.parameters(), lr=config.caption_lr, weight_decay=config.weight_decay)
    
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
    multilabel_criterion = nn.MultiLabelSoftMarginLoss()
    # multilabel_criterion = nn.MultiLabelMarginLoss()
    
    feature_extractor.train()
    caption_model.train()
    
    start_epoch, best_cider = 0, 0.0
    if config.resume_last:
        last_file_name = f"./save/ImageCaption-{config.feature_extractor}-{config.caption_model}/last.pth"
        data = torch.load(last_file_name)
        torch.set_rng_state(data['torch_rng_state'])
        torch.cuda.set_rng_state(data['cuda_rng_state'])
        np.random.set_state(data['numpy_rng_state'])
        random.setstate(data['random_rng_state'])
        feature_extractor.load_state_dict(data['feature_extractor'])
        caption_model.load_state_dict(data['caption_model'])
        optimizer.load_state_dict(data['optimizer'])
        start_epoch = data['current_epoch'] + 1
        best_cider = data['best_cider']
        
        if os.path.exists(last_file_name.replace('last.pth', 'best.pth')):
            best_data = torch.load(last_file_name.replace('last.pth', 'best.pth'))
            best_cider = best_data['best_cider']
        logging.info(f"Resuming from epoch {data['current_epoch']}, and best cider {best_cider}")
    
    logging.info("Start Training!")
    for epoch in range(start_epoch, config.end_epoch):
        if not config.use_rl:
            logging.info(f"Train - Epoch {epoch}")
            train_loss = train_xe(feature_extractor, caption_model, train_data_loader, cnn_optimizer, optimizer, criterion, multilabel_criterion, config)
            experiment.log({'train_loss': train_loss})
            logging.info(f"Train - Epoch {epoch}, Train Loss {train_loss :.3f}")
        else:
            train_scst()
        
        # Validation loss
        logging.info(f"Validation - Epoch {epoch}")
        val_loss = evaluate_loss(feature_extractor, caption_model, val_data_loader, criterion, multilabel_criterion, config)
        experiment.log({'val_loss': val_loss})
        logging.info(f"Validation - Epoch {epoch}, Validation Loss {val_loss :.3f}")
        
        # Validation scores
        logging.info(f"Validation score - Epoch {epoch}")
        score = evaluate_metrics(feature_extractor, caption_model, val_data_loader_bs_1, vocab, config)
        experiment.log({
            'v - BLEU-1': score['Bleu'][0], 'v - BLEU-2': score['Bleu'][1],
            'v - BLEU-3': score['Bleu'][2], 'v - BLEU-4': score['Bleu'][3],
            'v - METEOR': score['METEOR'], 'v - ROUGE_L': score['ROUGE_L'], 'v - CIDEr': score['CIDEr']
        })
        logging.info(f"Validation score - "
                     f"BLEU-1: {score['Bleu'][0] :.3f}, BLEU-2: {score['Bleu'][1] :.3f}, "
                     f"BLEU-3: {score['Bleu'][2] :.3f}, BLEU-4: {score['Bleu'][3] :.3f}, "
                     f"METEOR: {score['METEOR'] :.3f}, ROUGE_L: {score['ROUGE_L'] :.3f}, CIDEr: {score['CIDEr'] :.3f}")
        current_cider = score['CIDEr']
        
        # Test scores
        logging.info(f"Test score - Epoch {epoch}")
        score = evaluate_metrics(feature_extractor, caption_model, test_data_loader_bs_1, vocab, config)
        experiment.log({
            't - BLEU-1': score['Bleu'][0], 't - BLEU-2': score['Bleu'][1],
            't - BLEU-3': score['Bleu'][2], 't - BLEU-4': score['Bleu'][3],
            't - METEOR': score['METEOR'], 't - ROUGE_L': score['ROUGE_L'], 't - CIDEr': score['CIDEr']
        })
        logging.info(f"Test score - "
                     f"BLEU-1: {score['Bleu'][0] :.3f}, BLEU-2: {score['Bleu'][1] :.3f}, "
                     f"BLEU-3: {score['Bleu'][2] :.3f}, BLEU-4: {score['Bleu'][3] :.3f}, "
                     f"METEOR: {score['METEOR'] :.3f}, ROUGE_L: {score['ROUGE_L'] :.3f}, CIDEr: {score['CIDEr'] :.3f}")
        
        save_files = {
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'feature_extractor': feature_extractor.state_dict(),
            'caption_model': caption_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'current_epoch': epoch,
            'best_cider': best_cider,
            'vocab': vocab,
            'config': config
        }
        if not os.path.exists(f"./save/ImageCaption-{config.feature_extractor}-{config.caption_model}/"):
            os.mkdir(f"./save/ImageCaption-{config.feature_extractor}-{config.caption_model}/")
        if current_cider >= best_cider:
            best_cider = current_cider
            torch.save(save_files, f"./save/ImageCaption-{config.feature_extractor}-{config.caption_model}/best_{'MTL' if config['multi_task'] else 'wo MTL'}.pth")
            logging.info(f'Checkpoint {epoch} - best.pth saved!')
        
        torch.save(save_files, f"./save/ImageCaption-{config.feature_extractor}-{config.caption_model}/last_{'MTL' if config['multi_task'] else 'wo MTL'}.pth")
        logging.info(f'Checkpoint {epoch} - last.pth saved!')
    
    with torch.no_grad():
        # Test scores of best.pth
        feature_extractor.eval()
        caption_model.eval()
        best_file_name = f"./save/ImageCaption-{config.feature_extractor}-{config.caption_model}/best.pth"
        data = torch.load(best_file_name)
        logging.info(f"Load state from epoch: {data['current_epoch']} successfully")
        feature_extractor.load_state_dict(data['feature_extractor'])
        caption_model.load_state_dict(data['caption_model'])
        logging.info(f"Best Test score")
        best_score = evaluate_metrics(feature_extractor, caption_model, test_data_loader_bs_1, vocab, config)
        experiment.log({
            'BLEU-1': best_score['Bleu'][0], 'BLEU-2': best_score['Bleu'][1],
            'BLEU-3': best_score['Bleu'][2], 'BLEU-4': best_score['Bleu'][3],
            'METEOR': best_score['METEOR'], 'ROUGE_L': best_score['ROUGE_L'], 'CIDEr': best_score['CIDEr']
        })
        logging.info(f"Best Test score - "
                     f"BLEU-1: {best_score['Bleu'][0] :.3f}, BLEU-2: {best_score['Bleu'][1] :.3f}, "
                     f"BLEU-3: {best_score['Bleu'][2] :.3f}, BLEU-4: {best_score['Bleu'][3] :.3f}, "
                     f"METEOR: {best_score['METEOR'] :.3f}, ROUGE_L: {best_score['ROUGE_L'] :.3f}, CIDEr: {best_score['CIDEr'] :.3f}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    config = OmegaConf.load('config.yaml')
    main(config)
