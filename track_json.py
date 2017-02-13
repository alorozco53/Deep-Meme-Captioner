#!/usr/bin/env python
# -*- coding: utf-8
import json
import codecs
import sys

filename="items.json"

njson=0
nc=0
first=True
#  with open(filename) as f:
    #  nstack=0
    #  segment=""
    #  # Se desase del corchete
    #  while True:
    #      nc+=1
    #      c = f.read(1)
    #      if not c:
    #          break
    #      if not first:
    #          segment+=c
    #      first=False
    #      if c=='{':
    #          print c,nstack
    #          nstack+=1
    #      elif c=='}':
    #          print c,nstack
    #          nstack-=1
    #          if nstack==0:
    #              try:
    #                  memeinfo=json.loads(segment)
    #              except ValueError:
    #                  print segment
    #              njson+=1
    #              #HACER ALGO CON EL JSON
    #              segment=""
    #              first=True
 
nl=0
with open(filename) as f:
    for line in f:
        nl=+1
        line=line.strip()[:-1]
        if nl%100==0:
            print '.',

        if not first:
            try:
                memeinfo=json.loads(line)
                njson+=1
                if njson%1==0:
                    print '.',
            except ValueError:
                print line
                sys.exit(0)
        first=False



print "Número de jsons recuperados", njson
print "Número de líneas analizadas", nl

            
        

