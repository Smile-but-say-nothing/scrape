
def clean_sos_eos(raw_caption):
    cleaned_caption = []
    for item in raw_caption:
        if item == "<SOS>" or item == "<UNK>" or item == "<PAD>":
            continue
        elif item == "<EOS>":
            break
        else:
            cleaned_caption.append(item)
    return cleaned_caption

def clean_caption(token_list):
    remove_char = '[·’!"\#$%&\'()＃！（）*+,-./:;<=>?\@，：?￥★、…．＞【】［］《》？“”‘’\[\\]^_`{|}~。]+'
    return [token for token in token_list if token not in remove_char]