import torch
from PIL import Image
from utils import build_feature_extractor, build_caption_model, build_data_transform, clean_sos_eos, clean_caption
from itertools import groupby
import torch.nn.functional as F


def infer(src_path, weights_path):
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    data = torch.load(weights_path)
    config, vocab = data['config'], data['vocab']
    feature_extractor = build_feature_extractor(config.feature_extractor, pretrained=False, finetune=False)
    feature_extractor.to(device).load_state_dict(data['feature_extractor'])
    model_setting = getattr(config, config['caption_model'])
    vocab_size = len(vocab)
    caption_model = build_caption_model(vocab_size, config['caption_model'], model_setting, config)
    caption_model.to(device).load_state_dict(data['caption_model'])
    image = Image.open(src_path)
    transform = build_data_transform('val')
    image = transform(image)
    image = image.to(device).unsqueeze(0)
    att_feats, fc_feats = feature_extractor(image)
    # encode and decode
    sampled_ids = []
    if getattr(config, 'caption_model') in ['NIC', 'SoftAttention']:
        att_feats = caption_model.encoder(att_feats)
        sampled_ids = caption_model.decoder.sample(att_feats, max_len=50, search_mode="beam", beam_size=10)
    elif getattr(config, 'caption_model') in ['Transformer']:
        sampled_ids = caption_model.sample(image, att_feats, max_len=50, search_mode='beam', beam_size=10)
    h = clean_sos_eos([vocab.itos[x] for x in sampled_ids])
    h = [k for k, v in groupby(h)]
    h = ''.join(clean_caption([x for x in ' '.join(h)]))
    print(h)
    if config.multi_task:
        itol = {0: '平和质', 1: '气虚质', 2: '阳虚质', 3: '阴虚质', 4: '痰湿质',
                5: '湿热质', 6: '血瘀质', 7: '气郁质', 8: '特禀质'}
        last_logit = F.softmax(fc_feats, dim=1).squeeze()
        last_logit_sort = torch.sort(last_logit, descending=True)
        last_logit_sort_0, last_logit_sort_1 = last_logit_sort[0].tolist(), last_logit_sort[1].tolist()
        label_print = {str(itol[last_logit_sort_1[i]]): '%.3f' % (last_logit_sort_0[i]) for i in range(len(last_logit))}
        print(label_print)

if __name__ == '__main__':
    src_path = './Sha/陈华/0968 2022-09-08/背部一_seg.jpg'
    weights_path = './save/ImageCaption-resnet101-SoftAttention/best_MTL.pth'
    infer(src_path, weights_path)
