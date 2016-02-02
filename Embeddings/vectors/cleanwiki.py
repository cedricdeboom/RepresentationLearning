"""
Clean wikipedia dump
"""

__author__  = "Cedric De Boom"
__status__  = "beta"
__version__ = "0.2"
__date__    = "2015 June 18th"

import re
import sys


def clean(filename=''):
    text = False
    t1 = re.compile(r"<text ")
    t2 = re.compile(r"<\/text>")
    t3 = re.compile(r"==")
    r = re.compile(r"#redirect", re.IGNORECASE)

    with open(filename, 'r') as f:
        for line in f:
            if t1.search(line) is not None:
                text = True
                continue
            if r.search(line) is not None:
                text = False
            if text:
                line = line.lstrip()
                if t2.search(line) is not None:
                    text = False
                if line == '\n' or line.startswith(': ') or line.startswith('{|') or line.startswith('|') or line.startswith('!') or line.startswith('{{'):
                    continue
                if t3.search(line) is not None:
                    continue
                line = line.lower()
                line = re.sub(r"<.*>", '', line)
                line = re.sub(r"&amp;", '&', line)
                line = re.sub(r"&lt;", '<', line)
                line = re.sub(r"&gt;", '>', line)
                line = re.sub(r"<ref[^<]*<\/ref>", '', line)
                line = re.sub(r"<[^>]*>", '', line)
                line = re.sub(r"\[http:[^] ]*", '[', line)
                line = re.sub(r"\|thumb", '', line, re.IGNORECASE)
                line = re.sub(r"\|left", '', line, re.IGNORECASE)
                line = re.sub(r"\|right", '', line, re.IGNORECASE)
                line = re.sub(r"\|\d+px", '', line, re.IGNORECASE)
                line = re.sub(r"\[\[image:[^\[\]]*\|", '', line, re.IGNORECASE)
                line = re.sub(r"\[\[image:[^\[\]]*", '', line, re.IGNORECASE)
                line = re.sub(r"\[\[file:[^\[\]]*\|", '', line, re.IGNORECASE)
                line = re.sub(r"\[\[file:[^\[\]]*", '', line, re.IGNORECASE)
                line = re.sub(r"\[\[category:([^|\]]*)[^]]*\]\]", r"[[\1]]", line, re.IGNORECASE)
                line = re.sub(r"\[\[category:", '[[', line, re.IGNORECASE)
                line = re.sub(r"\[\[[a-z\-]*:[^\]]*\]\]", '', line)
                line = re.sub(r"\[\[[^\|\]]*\|", '[[', line)
                line = re.sub(r"{{[^}]*}}", '', line)
                line = re.sub(r"{{[^}]*}}", '', line)
                line = re.sub(r"{[^}]*}", '', line)
                line = re.sub(r"\[", '', line)
                line = re.sub(r"\]", '', line)
                line = re.sub(r"&[^;]*;", ' ', line)

                line = re.sub(r"[-+]?[\d,]*\.*\d+", '0', line)
                line = re.sub(r"[^\w]", ' ', line)
                line = re.sub(r"\s+", ' ', line)
                line = line.strip()

                if len(line) > 1 and not line.startswith('__'):
                    if text:
                        sys.stdout.write(line + ' ')
                    else:
                        sys.stdout.write(line + '\n')

if __name__ == '__main__':
    clean(sys.argv[1])