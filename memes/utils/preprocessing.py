#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import codecs
import shutil
import argparse

from utils.data_utils import clean_caption

def _next_interval(curr_low, curr_up, delta):
    return curr_low + delta, curr_up + delta

def cut_dataset(dataset_path,
                max_captions,
                dest_path,
                prefix='part-{}-to-{}'):
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
    if not os.path.isdir(dest_path):
        os.mkdir(dest_path)
    caption_count = 0
    lower_bound = 0
    upper_bound = max_captions
    current_dir = os.path.join(dest_path,
                               prefix.format(lower_bound, upper_bound))
    os.mkdir(current_dir)
    vocab = {}
    word_count = 1
    for meme in dataset_contents:
        # Create a necessary directory
        if caption_count > 0 and caption_count % max_captions == 0:
            lower_bound, upper_bound = _next_interval(lower_bound,
                                                      upper_bound,
                                                      max_captions)
            current_dir = os.path.join(dest_path,
                                       prefix.format(lower_bound, upper_bound))
            os.mkdir(current_dir)

        old_meme_file = os.path.join(dataset_path, meme, '{}.csv'.format(meme))
        if os.path.exists(old_meme_file):
            # Create the new file paths and copy the image and metadata to the new dir
            old_img_path = os.path.join(dataset_path, meme, '{}.jpg'.format(meme))
            old_metadata = os.path.join(dataset_path, meme, '{}_metadata.csv'.format(meme))
            new_meme_file = os.path.join(current_dir, meme, '{}.csv'.format(meme))
            new_img_path = os.path.join(current_dir, meme, '{}.jpg'.format(meme))
            new_metadata = os.path.join(current_dir, meme, '{}_metadata.csv'.format(meme))
            os.mkdir(os.path.join(current_dir, meme))
            shutil.copyfile(old_img_path, new_img_path)
            shutil.copyfile(old_metadata, new_metadata)
            with codecs.open(old_meme_file, 'r') as f:
                print('Reading', old_meme_file, '...')
                reader = csv.reader(f)
                first = True
                data = []
                for row in reader:
                    data.append(row)
                    # Update vocabulary
                    if not first:
                        cap = clean_caption(row[0])
                        if cap:
                            for w in cap.split():
                                try:
                                    vocab[w] = vocab[w]
                                except KeyError:
                                    vocab[w] = word_count
                                    word_count += 1
                            caption_count = caption_count + 1\
                                            if not first else caption_count
                    first = False
                    if caption_count > 0 and caption_count % max_captions == 0:
                        # Save captions because we've reached a limit
                        if data:
                            with codecs.open(new_meme_file, 'w') as nf:
                                print('Writing in', new_meme_file, '...')
                                writer = csv.writer(nf)
                                writer.writerows(data)
                            data = [data[0]]
                        else:
                            data = []
                        # Create the new file paths and copy the image and metadata to the new dir
                        lower_bound, upper_bound = _next_interval(lower_bound,
                                                                  upper_bound,
                                                                  max_captions)
                        current_dir = os.path.join(dest_path,
                                                   prefix.format(lower_bound, upper_bound))
                        os.mkdir(current_dir)
                        new_meme_file = os.path.join(current_dir, meme, '{}.csv'.format(meme))
                        new_img_path = os.path.join(current_dir, meme, '{}.jpg'.format(meme))
                        new_metadata = os.path.join(current_dir, meme, '{}_metadata.csv'.format(meme))
                        os.mkdir(os.path.join(current_dir, meme))
                        shutil.copyfile(old_img_path, new_img_path)
                        shutil.copyfile(old_metadata, new_metadata)

                # Save captions because we're done with a meme
                if data:
                    with codecs.open(new_meme_file, 'w') as nf:
                        print('Writing in', new_meme_file, '...')
                        writer = csv.writer(nf)
                        writer.writerows(data)

    # Create vocabulary file
    vocab_file = os.path.join(dest_path, 'word_count.txt')
    with codecs.open(vocab_file, 'w') as f:
        lines = ['{} {}\n'.format(k, v) for k, v in vocab.items()]
        lines.append('</S> {}\n'.format(len(lines) - 1))
        lines.append('<S> {}\n'.format(len(lines) - 1))
        print('Writing vocabulary file in', vocab_file)
        f.writelines(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script\'s argument parser')
    parser.add_argument('old_meme_dir', help='directory where current dataset is stored')
    parser.add_argument('new_meme_dir', help='new directory where subdatasets will be stored')
    parser.add_argument('k', help='maximum number of captions per subdataset',
                        type=int)
    parser.add_argument('-p', help='prefix for new subdataset',
                        default='part-{}-to-{}')
    args = parser.parse_args()
    cut_dataset(args.old_meme_dir, args.k,
                args.new_meme_dir, args.p)
