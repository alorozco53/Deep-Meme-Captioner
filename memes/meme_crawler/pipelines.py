#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import csv
import os

class CSVCaptionsPipeline(object):

    METADATA_HEADER = ['meme_name', 'meme_img_url']
    CAPTIONS_HEADER = ['up_caption', 'down_caption', 'language']

    def process_item(self, item, spider):
        if not os.path.exists('meme_characters'):
            os.mkdir('meme_characters')
        if spider.name == 'memecaptionspider':
            self.item = item
            self.meme_path = os.path.join('meme_characters',
                                          self.item['meme_name'])
            print('serializing data for', self.item['meme_name'], '...')
            # create directory, if needed
            if not os.path.exists(self.meme_path):
                os.mkdir(self.meme_path)
            self.save_image()
            self.save_captions()
            print('successfully serialized', self.item['meme_name'])

    def save_image(self):
        # save plain meme image
        meme_filepath = os.path.join(self.meme_path,
                                     '{}.jpg'.format(self.item['meme_name']))
        if not os.path.isfile(meme_filepath):
            os.system('wget ' +\
                      '--accept .jpg,.jpeg ' +\
                      '--cookies=on ' +\
                      '-p \"' + self.item['meme_img_url']  + '\" ' +\
                      '-O \"' + meme_filepath + '\"')

    def save_captions(self):
        # write metadata, if needed
        meta_path = os.path.join(self.meme_path,
                                 '{}_metadata.csv'.format(self.item['meme_name']))
        if not os.path.exists(meta_path) or os.path.getsize(meta_path) <= 0:
            with codecs.open(meta_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(self.METADATA_HEADER)
                writer.writerow((self.item['meme_name'],
                                 self.item['meme_img_url']))
        # write the captions
        caption_path = os.path.join(self.meme_path,
                                    '{}.csv'.format(self.item['meme_name']))
        if not os.path.exists(caption_path) or os.path.getsize(caption_path) <= 0:
            with codecs.open(caption_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(self.CAPTIONS_HEADER)
                data = []
                for caption in self.item['meme_captions']:
                    data.append(caption['up_caption'])
                    data.append(caption['down_caption'])
                    # data.append(caption['img_url'])
                    data.append(caption['language'])
                    print('data', data)
                    writer.writerow(data)
                    data = []
        else:
            captions = None
            with codecs.open(caption_path, 'r') as f:
                captions = [f'{uc} {dc}' for uc, dc, _ in csv.reader(f)][1:]
            with codecs.open(caption_path, 'a+') as f:
                writer = csv.writer(f)
                for caption in self.item['meme_captions']:
                    ucap = caption['up_caption']
                    dcap = caption['down_caption']
                    cap = f'{ucap} {dcap}'
                    if cap not in captions:
                        data = [ucap, dcap]
                        # data.append(caption['img_url'])
                        data.append(caption['language'])
                        writer.writerow(data)
                        print('data', data)
