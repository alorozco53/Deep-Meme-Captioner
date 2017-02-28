#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import codecs
import argparse
import os
import csv

GEN_PATH = None
METADATA_HEADER = ['meme_name', 'meme_img_url']
CAPTIONS_HEADER = ['caption', 'img_url', 'language']

def to_csv(meme_info):
    meme_path = os.path.join(GEN_PATH, meme_info['meme_name'])
    # create directory, move image, if needed
    if not os.path.exists(meme_path):
        os.mkdir(meme_path)
    img_path = os.path.join(GEN_PATH, meme_info['meme_name'] + '.jpg')
    if not os.path.exists(img_path) or os.path.getsize(img_path) <= 0:
        try:
            img_old_path = os.join(GEN_PATH, meme_info['meme_name'] + '.jpg')
            os.rename(img_old_path, img_path)
        except:
            print('', end='', flush=True)

    # write metadata, if needed
    meta_path = os.path.join(meme_path,
                             meme_info['meme_name'] + '_metadata.csv')
    if not os.path.exists(meta_path) or os.path.getsize(meta_path) <= 0:
        with codecs.open(meta_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(METADATA_HEADER)
            writer.writerow((meme_info['meme_name'],
                             meme_info['meme_img_url']))

    # write the captions
    caption_path = os.path.join(meme_path,
                                meme_info['meme_name'] + '.csv')
    if not os.path.exists(caption_path) or os.path.getsize(caption_path) <= 0:
        with codecs.open(caption_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(CAPTIONS_HEADER)
            data = []
            for caption in meme_info['meme_captions']:
                data.append(caption['caption'])
                data.append(caption['img_url'])
                data.append(caption['language'])
                writer.writerow(data)
                data = []
    else:
        captions = None
        with codecs.open(caption_path, 'r') as f:
            captions = [row[0] for row in csv.reader(f)][1:]
        with codecs.open(caption_path, 'a+') as f:
            writer = csv.writer(f)
            for caption in meme_info['meme_captions']:
                cap = caption['caption']
                if cap not in captions:
                    data = [cap]
                    data.append(caption['img_url'])
                    data.append(caption['language'])
                    writer.writerow(data)


def parse(filepath):
    first = True
    with codecs.open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line[-1] == ',':
                line = line.strip()[:-1]
            if not first:
                memeinfo = json.loads(line)
                to_csv(memeinfo)
                print('. ', end='', flush=True)
            first = False

def main():
    global GEN_PATH
    parser = argparse.ArgumentParser(description='script\'s argument parser')
    parser.add_argument('filepath', help='pseudo-json file to be parsed')
    parser.add_argument('meme_characs_path', help='directory where memes are stored')
    args = parser.parse_args()
    GEN_PATH = args.meme_characs_path
    parse(args.filepath)

if __name__ == '__main__':
    main()
