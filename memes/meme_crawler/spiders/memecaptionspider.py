#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from scrapy.spiders import Spider
from meme_crawler.items import MemeCrawlerItem, CaptionItem
from scrapy.http import Request

class MemeCaptionSpider(Spider):
    name = 'memecaptionspider'
    allowed_domains = ['memegenerator.net']
    start_urls = ['https://memegenerator.net/memes/popular/alltime/page/1']# ,
                  # 'https://memegenerator.net/memes/popular/alltime/page/10',
                  # 'https://memegenerator.net/memes/popular/alltime/page/50',
                  # 'https://memegenerator.net/memes/popular/alltime/page/100']
    MAX_NUMBER_OF_MEMES = 20
    MAX_NUMBER_OF_PAGES = 10

    def parse(self, response):
        # get current meme-list page number
        try:
            meme_list_page = response.meta['list_page']
        except KeyError:
            meme_list_page = 1

        # get a list for all memes in the current webpage
        meme_names = response.xpath('//span[@class="display-name"]/text()').extract()
        plain_memes = response.xpath('//img[@class="mr10px loading-bg"]/@src').extract()
        if not meme_names:
            return
        meme_name_map = [(meme_names[i], plain_memes[i]) for i in range(len(plain_memes))]
        for name, url in meme_name_map:
            # process the name to be used as url and directory name
            meme_name = name.strip().lower().replace(' ', '-')
            meme_url = 'https://memegenerator.net/{}/images/popular/alltime/page/1'.format(meme_name)
            # create and yield request to meme_url
            request = Request(meme_url, callback=self.parse_memes)
            request.meta['counter'] = 1
            request.meta['meme_img_url'] = url
            request.meta['meme_name'] = meme_name
            request.meta['meme_page'] = 1
            yield request

        # go to next page
        try:
            overlimit = response.meta['page_counter'] > self.MAX_NUMBER_OF_PAGES
        except KeyError:
            overlimit = False
        if not overlimit:
            counter = response.meta['page_counter'] if 'page_counter' in response.meta.keys() else 0
            next_pages = [w for w in response.xpath('//li/a/@href').extract() if 'page/' in w][1:]
            for next_page in next_pages:
                next_page_suffix = next_page[next_page.rfind('/') + 1 : ]
                if int(next_page_suffix) == meme_list_page + 1:
                    request = Request('https://memegenerator.net{}'.format(next_page),
                                      callback=self.parse)
                    request.meta['list_page'] = meme_list_page + 1
                    counter += 1
                    request.meta['page_counter'] = counter
                    yield request
                    break

    def parse_memes(self, response):
        meme_img_url = response.meta['meme_img_url']
        meme_name = response.meta['meme_name']
        counter = response.meta['counter']
        meme_page = response.meta['meme_page']
        try:
            item = response.meta['item']
        except KeyError:
            item = MemeCrawlerItem()
            item['meme_name'] = meme_name
            item['meme_img_url'] = meme_img_url
            item['meme_captions'] = []

        img_urls = [im for im in response.xpath('//img/@src').extract()
                    if '.jpg' in im and '250x250' in im]
        up_captions = response.xpath('//div[@class="only-above-768"]//div[@class="optimized-instance-text0"]/text()').extract()
        down_captions = response.xpath('//div[@class="only-above-768"]//div[@class="optimized-instance-text1"]/text()').extract()
        cap_img_map = zip(up_captions, down_captions)

        # process each meme
        # print('cap_img_map', list(cap_img_map))
        # print(len(up_captions), len(down_captions), len(img_urls))
        for up_caption, down_caption in cap_img_map:
            cap_item = CaptionItem()
            cap_item['up_caption'] = up_caption.strip()
            cap_item['down_caption'] = down_caption.strip()
            # cap_item['img_url'] = img_url
            cap_item['language'] = response.xpath('//meta[@name="language"]/@content').extract()[0]
            item['meme_captions'].append(cap_item)
            counter += 1

        # check if the limit for downloaded memes has been reached
        # if counter > self.MAX_NUMBER_OF_MEMES:
        print(item['meme_captions'])
        yield item

        # go to the next page
        next_pages = [w for w in response.xpath('//li/a/@href').extract() if 'page/' in w][1:]
        for next_page in next_pages:
            next_page_suffix = next_page[next_page.rfind('/') + 1 : ]
            if int(next_page_suffix) == meme_page + 1:
                request = Request('https://memegenerator.net' + next_page,
                                  callback=self.parse_memes)
                request.meta['meme_name'] = meme_name
                request.meta['counter'] = counter
                request.meta['meme_img_url'] = meme_img_url
                request.meta['meme_page'] = meme_page + 1
                request.meta['item'] = item
                yield request
                break
