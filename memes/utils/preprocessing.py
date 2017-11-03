#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import re
import codecs
import shutil
import argparse

from random import shuffle

HEADER = ['MEME_CAPTION', 'IMG_URL', 'LANGUAGE', 'IMG_PATH']


# Some useful regexes
symbol_re = re.compile('[^\w\s\']')
space_re = re.compile('[\s\t\n]{2,}')

def clean_caption(caption):
    # Remove non-alphanumeric chars
    cleaned = re.sub(symbol_re, '', caption.lower())

    # Remove unnecessary spaces
    cleaned = re.sub(space_re, ' ', cleaned)

    return cleaned.strip()

def _next_interval(curr_low, curr_up, delta):
    return curr_low + delta, curr_up + delta

def cut_dataset(dataset_path,
                max_captions,
                dest_path,
                img_dest_path=None,
                prefix='part-{}-to-{}',
                threshold=5):
    """Cuts the dataset into the necessary parts, such
    that each part has at most max_captions captions.

    Args:
      :dataset_path, string: path to the meme dataset
      :max_captions, int > 0: maximum number of captions per
      part
      :dest_path, string: path where subdatasets will be saved
      :prefix, string: prefix of directories which will contain each subdataset
    """
    assert max_captions > 0

    dataset_contents = [m for m in os.listdir(dataset_path)
                        if os.path.isdir(os.path.join(dataset_path, m))
                        and os.listdir(os.path.join(dataset_path, m))]
    vocab = {}
    data = []
    for meme in dataset_contents:
        old_meme_file = os.path.join(dataset_path, meme, '{}.csv'.format(meme))
        if os.path.exists(old_meme_file):
            with codecs.open(old_meme_file, 'r') as f:
                print('Reading', old_meme_file, '...')
                reader = csv.reader(f)
                first = True
                for row in reader:
                    # Update vocabulary
                    if not first:
                        cap = clean_caption(row[0])
                        img_url = row[1]
                        language = row[2]
                        if cap:
                            c = cap.split()
                            data.append([meme, c, img_url, language])
                            for w in c:
                                try:
                                    vocab[w] += 1
                                except KeyError:
                                    vocab[w] = 1
                    first = False


    # More cleaning!
    slit = lambda w: w if vocab[w] > threshold else '<UNK>'
    data = [[m, list(map(slit, c)), i, l] for m, c, i, l in data]
    unk_count = sum([c.count('<UNK>') for _, c, _, _ in data ])

    # Shuffle list randomly
    print('Shuffling memes randomly!')
    shuffle(data)

    # Destination path
    if not os.path.isdir(dest_path):
        os.mkdir(dest_path)

    # Useful variables
    caption_count = 0
    lower_bound = 0
    upper_bound = max_captions

    # Directory for memes to be saved in
    current_dir = os.path.join(dest_path,
                               prefix.format(lower_bound, upper_bound))
    os.mkdir(current_dir)

    # Images' destination path
    assert not img_dest_path or (img_dest_path and os.path.isdir(img_dest_path))
    if not img_dest_path:
        imdp = os.path.join(dest_path, 'img/')
        os.mkdir(imdp)
    else:
        imdp = img_dest_path

    # Save memes accordingly
    csv_file = os.path.join(current_dir, 'memes.csv')
    f = codecs.open(csv_file, 'w')
    writer = csv.writer(f)
    writer.writerow(HEADER)
    for meme, caption, img_url, language in data:
        print('Saving data in', csv_file, '...')
        # Change directory if needed
        if caption_count > 0 and caption_count % max_captions == 0:
            lower_bound, upper_bound = _next_interval(lower_bound,
                                                      upper_bound,
                                                      max_captions)
            current_dir = os.path.join(dest_path,
                                       prefix.format(lower_bound, upper_bound))
            os.mkdir(current_dir)
            csv_file = os.path.join(current_dir, 'memes.csv')
            f.close()
            f = codecs.open(csv_file, 'w')
            writer = csv.writer(f)
            writer.writerow(HEADER)

        # Save data
        img_path = os.path.join(imdp, '{}.jpg'.format(meme))
        c = ' '.join(caption)
        writer.writerow([c, img_url, language, img_path])

        # Save images
        old_img_path = os.path.join(dataset_path, meme, '{}.jpg'.format(meme))
        if not os.path.isfile(img_path):
            shutil.copyfile(old_img_path, img_path)

        # Increase counter
        caption_count += 1


    # Create vocabulary file
    vocab_file = os.path.join(dest_path, 'word_count.txt')
    with codecs.open(vocab_file, 'w') as f:
        print('Writing vocabulary file in', vocab_file)
        for k, v in vocab.items():
            if v > threshold:
                f.write('{} {}\n'.format(k, v))
        f.write('</S> {}\n'.format(len(vocab.keys()) - 1))
        f.write('<S> {}\n'.format(len(vocab.keys()) - 1))
        f.write('<UNK> {}\n'.format(unk_count))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script\'s argument parser')
    parser.add_argument('old_meme_dir', help='directory where current dataset is stored')
    parser.add_argument('new_meme_dir', help='new directory where subdatasets will be stored')
    parser.add_argument('k', help='maximum number of captions per subdataset',
                        type=int)
    parser.add_argument('-p', help='prefix for new subdataset',
                        default='part-{}-to-{}')
    parser.add_argument('-t', help='word occurrency threshold',
                        default=5, type=int)
    parser.add_argument('-i', help='directory where all images will be stored',
                        default=None)
    args = parser.parse_args()
    cut_dataset(args.old_meme_dir,
                args.k,
                args.new_meme_dir,
                args.i,
                args.p,
                args.t)
