#! /bin/python3

import sys

fd = sys.stdin

for line in fd:
    s = line.split()
    if len(s) > 6:
        if s[1] == '-LRB-':
            s[1] = '('
            s[5] = '$('
        elif s[1] == '-RRB-':
            s[1] = ')'
            s[5] = '$('
        print(s[1] + "/" + s[5], end=' ')
    else:
        print()

