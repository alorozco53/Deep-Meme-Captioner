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

def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

def stats(GEN_PATH, verbose=True):
    words = 0
    text = ''
    meme_characters = 0
    captions = 0
    for meme in os.listdir(GEN_PATH):
        meme_csv = os.path.join(GEN_PATH, meme, '{}.csv'.format(meme))
        if os.path.exists(meme_csv) and os.path.getsize(meme_csv) > 0:
            if verbose:
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
    return captions, meme_characters, words, len(list(set(text))), len(text.encode('utf-8'))


def main():
    parser = argparse.ArgumentParser(description='script\'s argument parser')
    parser.add_argument('meme_characs_path', help='directory where memes are stored')
    parser.add_argument('-v', help='verbose mode', action='store_true')
    args = parser.parse_args()
    ncaptions, nmeme_characters, nwords, nchars, total_size = stats(args.meme_characs_path, args.v)
    print('\n\ntotal number of captions', ncaptions)
    print('total number of meme characters', nmeme_characters)
    print('total number of words:', nwords)
    print('total number of characters:', nchars)
    print('total size:', sizeof_fmt(total_size))


if __name__ == '__main__':
    main()
