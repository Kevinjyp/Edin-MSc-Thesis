import argparse
import os.path
import pdb
import pandas as pd

def sentence_length(args):
    lengths = []
    with open(args.input, 'r') as f:
        lines = f.readlines()
        for line in lines:
            lengths.append(len(line.replace('\n', '')))
    df_lengths = pd.Series(lengths)
    print(df_lengths.describe())
    st = df_lengths.describe()
    with open('./stat.out', 'w') as f:
        f.write('\t'.join([str(s) for i,s in enumerate(st) if i>0]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode")
    parser.add_argument("-i", "--input")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()

    if args.mode == 'sl':
        sentence_length(args)