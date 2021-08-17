#!/usr/bin/env python3

import argparse
import os
import os.path
import sys

# import lxml.etree as ET
from bs4 import BeautifulSoup


def sgm(args):
    with open(args.input, 'r') as f:
        soup = BeautifulSoup(f, 'html.parser')
        # import pdb; pdb.set_trace()
        texts = [s.get_text() for s in soup.find_all('seg')]
    with open(args.output, 'w') as f:
        f.writelines("%s\n" % t for t in texts)

def main(args):
    output_stem = args.output_stem
    if output_stem == None:
        output_stem = args.xml_file[:-4]

    pair = args.xml_file.split(".")[-2]
    src, tgt = pair.split("-")

    tree = ET.parse(args.xml_file)
    root = tree.getroot()

    source_texts, target_texts = [], []
    for element in root:
        for all_tags in element.findall('.//'):
            if 'source' in str(all_tags.tag):
                source_texts.append(all_tags.text)
            if 'target' in str(all_tags.tag):
                target_texts.append(all_tags.text)

    with open(output_stem + "." + src, "w") as ofh:
        ofh.writelines("%s\n" % l for l in source_texts)

    with open(output_stem + "." + tgt, "w") as ofh:
        ofh.writelines("%s\n" % l for l in target_texts)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()
    # main(args)
    sgm(args)
