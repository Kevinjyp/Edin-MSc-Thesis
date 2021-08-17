#!/bin/bash

python3 ~/yunpengjiao/repos/MUSE/unsupervised.py --src_lang ha --tgt_lang de --src_emb ha/data/corpus.ha.tok.emb.vec.full --tgt_emb $1 --n_refinement 5 --emb_dim 512 --cuda 1  --dis_most_frequent 0 --export txt --dico_eval /home/yunpengjiao/yunpengjiao/mscproject/monolingual/de-ha/de-ha.ft.align