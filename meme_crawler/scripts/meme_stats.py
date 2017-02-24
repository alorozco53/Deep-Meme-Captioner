#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import codecs
import sys
import argparse

GEN_PATH = None

def stats():
    words = 0
    text = ''
    meme_characters = 0
    captions = 0
    for meme in os.listdir(GEN_PATH):
        meme_csv = os.path.join(GEN_PATH, meme, '{}.csv'.format(meme))
        if os.path.exists(meme_csv) and os.path.getsize(meme_csv) > 0:
            print('getting info for', meme, 'meme...')
            meme_characters += 1
            with codecs.open(meme_csv, 'r') as f:
                reader = csv.reader(f)
                first = True
                for row in reader:
                    if not first:
                        text += '{} '.format(row[0])
                        words += len(row[0].split())
                        captions += 1
                    first = False
    print('\n\ntotal number of captions', captions)
    print('total number of meme characters', meme_characters)
    print('total number of words:', words)
    print('total number of characters:', len(list(set(text))))
    print('total size:', sys.getsizeof(text), 'bytes')



def main():
    global GEN_PATH
    parser = argparse.ArgumentParser(description='script\'s argument parser')
    parser.add_argument('meme_characs_path', help='directory where memes are stored')
    args = parser.parse_args()
    GEN_PATH = args.meme_characs_path
    stats()

if __name__ == '__main__':
    main()
