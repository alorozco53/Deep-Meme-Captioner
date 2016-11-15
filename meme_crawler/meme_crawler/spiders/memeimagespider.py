#!/usr/bin/python
# -*- coding: utf-8 -*-

import os

from scrapy.spiders import Spider
from scrapy.http import Request

class MemeImageSpider(Spider):
    name = 'memeimagespider'
    allowed_domains = ['memegenerator.net']
    start_urls = ['https://memegenerator.net/memes/popular/alltime']
    MAX_NUMBER_OF_MEMES = 1500

    def parse(self, response):
        # get current meme-list page number
        try:
            meme_list_page = response.meta['list_page']
        except KeyError:
            meme_list_page = 1

        # get a list for all memes in the current webpage and process it conveniently
        meme_names = response.xpath('//strong[@class="generator-name"]/text()').extract()

        # crawl all memes per recognized name
        for name in meme_names:
            # process the name to be used as url and directory name
            meme_name = name.strip().lower().replace(' ', '-')
            meme_url = 'https://memegenerator.net/' + meme_name


            # create directory to store meme images
            if not os.path.exists('meme_images/'):
                os.mkdir('meme_images')
            if not os.path.exists('meme_images/' + meme_name + '/'):
                os.mkdir('meme_images/' + meme_name + '/')

            # create and yield request to meme_url
            request = Request(meme_url, callback=self.parse_meme_mainpage)
            request.meta['meme_path'] = 'meme_images/' + meme_name + '/'
            yield request

        # go to next page
        next_pages = [w for w in response.xpath('//li/a/@href').extract() if 'page/' in w]
        for next_page in next_pages:
            next_page_suffix = next_page[next_page.rfind('/') + 1 : len(next_page)]
            if int(next_page_suffix) == meme_list_page + 1:
                request = Request('https://memegenerator.net' + next_page,
                                  callback=self.parse)
                request.meta['list_page'] = meme_list_page + 1
                yield request
                break

    def parse_meme_mainpage(self, response):
        '''
        This method is called only once per meme
        '''
        meme_path = response.meta['meme_path']

        # check for last meme (images) saved in disk
        numb_exist_memes = len([name for name in os.listdir(meme_path)\
                                if '.jpg' in name])
        counter = numb_exist_memes + 1
        meme_page = int(float(numb_exist_memes) / 15.0 + 1.0)
        url = response.url + '/images/popular/alltime/page/' + str(meme_page)

        # build a request to start saving for the next meme in the order they appear
        # in the website
        request = Request(url, callback=self.parse_memes)
        request.meta['counter'] = counter
        request.meta['meme_page'] = meme_page
        request.meta['meme_path'] = meme_path
        yield request

    def parse_memes(self, response):
        meme_path = response.meta['meme_path']
        counter = response.meta['counter']
        meme_page = response.meta['meme_page']
        meme_urls = response.xpath('//img[@class="item-image"]/@src').extract()

        # check if the limit for downloaded memes has been reached
        if counter > self.MAX_NUMBER_OF_MEMES:
            return

        # process each meme
        for img_url in meme_urls:
            # save image to local disk
            if not os.path.isfile(meme_path + str(counter) + '.jpg'):
                os.system('wget ' +\
                          '--accept .jpg,.jpeg ' +\
                          '--cookies=on ' +\
                          '-p \"' + img_url  + '\" ' +\
                          '-O \"' + meme_path + str(counter) + '.jpg\"')
                counter += 1

        # go to next page
        next_pages = [w for w in response.xpath('//li/a/@href').extract() if 'page/' in w]
        for next_page in next_pages:
            next_page_suffix = next_page[next_page.rfind('/') + 1 : len(next_page)]
            if int(next_page_suffix) == meme_page + 1:
                request = Request('https://memegenerator.net' + next_page,
                                  callback=self.parse_memes)
                request.meta['meme_path'] = meme_path
                request.meta['counter'] = counter
                request.meta['meme_page'] = meme_page + 1
                yield request
                break
