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


def reduce(csvpath, destpath, k):
    assert os.path.exists(csvpath) and os.path.exists(destpath)
    assert k > 0
    meme, extension = os.path.splitext(os.path.basename(csvpath))
    output = []
    print(meme)
    # Read csv source file
    with codecs.open(csvpath, 'r') as f:
        reader = csv.reader(f)
        count = 0
        output = []
        for line in reader:
            output.append(line)
            count += 1
            if count >= k:
                break

    # Write in destination file
    destfile = os.path.join(destpath, meme + extension)
    with codecs.open(destfile, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script\'s argument parser')
    parser.add_argument('old_meme_dir', help='directory where current dataset is stored')
    parser.add_argument('new_meme_dir', help='new directory where subdatasets will be stored')
    parser.add_argument('k', help='maximum number of captions per subdataset',
                        type=int)
    args = parser.parse_args()

    # Create new dir if needed
    if not os.path.exists(args.new_meme_dir):
        os.mkdir(args.new_meme_dir)

    # Loop and move needed files
    for dirpath, dirnames, filenames in os.walk(args.old_meme_dir):
        if not dirnames:
            meme = os.path.basename(dirpath)
            destpath = os.path.join(args.new_meme_dir, meme)
            if not os.path.exists(destpath):
                os.mkdir(destpath)
            for f in filenames:
                origin = os.path.join(dirpath, f)
                if not f.endswith('.csv') or '_metadata' in f:
                    dest = os.path.join(destpath, f)
                    shutil.copy(origin, dest)
                else:
                    reduce(origin, destpath, args.k)
