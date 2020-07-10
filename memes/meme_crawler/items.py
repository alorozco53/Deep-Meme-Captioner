#!/usr/bin/python
# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class MemeCrawlerItem(scrapy.Item):
    meme_name = scrapy.Field()
    meme_img_url = scrapy.Field()
    meme_captions = scrapy.Field()
    pass

class CaptionItem(scrapy.Item):
    up_caption = scrapy.Field()
    down_caption = scrapy.Field()
    img_url = scrapy.Field()
    language = scrapy.Field()
    pass
