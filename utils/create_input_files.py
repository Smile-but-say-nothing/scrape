import json
from collections import Counter
import os
import h5py
from tqdm import tqdm

with open('./dataset_sha.json', 'r', encoding='gbk') as j:
    data = json.load(j)

train_image_paths = []
train_image_captions = []
val_image_paths = []
val_image_captions = []
test_image_paths = []
test_image_captions = []
word_freq = Counter()
min_word_freq = 1

for e in data['annotations']:
    captions = []
    for t in e['sentences']:
        # Update word frequency
        word_freq.update(t['tokens'])
        captions.append(t['tokens'])

    path = os.path.join(e['file_path'], e['file_name'])

    if e['split'] in {'Train'}:
        train_image_paths.append(path)
        train_image_captions.append(captions)
    elif e['split'] in {'Val'}:
        val_image_paths.append(path)
        val_image_captions.append(captions)
    elif e['split'] in {'Test'}:
        test_image_paths.append(path)
        test_image_captions.append(captions)

# Create word map
words = [w for w in word_freq.keys() if word_freq[w] >= min_word_freq]
word_map = {k: v + 1 for v, k in enumerate(words)}
word_map['<UNK>'] = len(word_map) + 1
word_map['<SOS>'] = len(word_map) + 1
word_map['<EOS>'] = len(word_map) + 1
word_map['<PAD>'] = 0

captions_per_image = 1
# Create a base/root name for all output files
base_filename = str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'
with open(os.path.join('WORDMAP_' + base_filename + '.json'), 'w', encoding='gbk') as j:
    json.dump(word_map, j)

for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                               (val_image_paths, val_image_captions, 'VAL'),
                               (test_image_paths, test_image_captions, 'TEST')]:

    with h5py.File(os.path.join(split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
        # Make a note of the number of captions we are sampling per image
        h.attrs['captions_per_image'] = captions_per_image

        # Create dataset inside HDF5 file to store images
        images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

        print("\nReading %s images and captions, storing to file...\n" % split)

        enc_captions = []
        caplens = []

        for i, path in enumerate(tqdm(impaths)):

            # Sample captions
            if len(imcaps[i]) < captions_per_image:
                captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
            else:
                captions = sample(imcaps[i], k=captions_per_image)

            # Sanity check
            assert len(captions) == captions_per_image

            # Read images
            img = imread(impaths[i])
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
                img = np.concatenate([img, img, img], axis=2)
            img = imresize(img, (256, 256))
            img = img.transpose(2, 0, 1)
            assert img.shape == (3, 256, 256)
            assert np.max(img) <= 255

            # Save image to HDF5 file
            images[i] = img

            for j, c in enumerate(captions):
                # Encode captions
                enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                    word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                # Find caption lengths
                c_len = len(c) + 2

                enc_captions.append(enc_c)
                caplens.append(c_len)

        # Sanity check
        assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

        # Save encoded captions and their lengths to JSON files
        with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
            json.dump(enc_captions, j)

        with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
            json.dump(caplens, j)
    