#!/bin/python3

import sys, re

args = sys.argv

print(args)
assert len(args) == 4

merge = args[1]
source1 = args[2]
source2 = args[3]

BOS = re.compile(r'^#BOS\s+([0-9]+)')
EOS = re.compile(r'^#EOS\s+([0-9]+)$')

S2 = '-'

def skip_sent(f):
    begin = False
    while True:
        line = f.readline()
        if BOS.search(line):
            begin = True
        if EOS.search(line) and begin:
            break

def cat_sent(f):
    begin = False
    end = False
    while True:
        line = f.readline()
        if BOS.search(line):
            begin = True
        if EOS.search(line) and begin:
            end = True
        if begin:
            print(line, end='')
        if end:
            break

with open(merge) as mf, open(source1) as sf1, open(source2) as sf2:
    results = mf.readline()
    for i, select in enumerate(results):
        if select == '\n':
            continue
        print(i + 1, file=sys.stderr)
        if select == S2:
            cat_sent(sf2)
            skip_sent(sf1)
            print()
        else:
            cat_sent(sf1)
            skip_sent(sf2)
            print()

