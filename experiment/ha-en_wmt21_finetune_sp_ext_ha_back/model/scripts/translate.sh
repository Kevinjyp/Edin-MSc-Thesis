#!/bin/bash
set -eof pipefail

. `dirname $0`/vars

MODEL=$working_dir/model.npz.best-translation.npz

$moses_scripts/tokenizer/normalize-punctuation.perl -l $src \
  | $moses_scripts/tokenizer/tokenizer.perl -l $src -q \
  | $moses_scripts/recaser/truecase.perl --model $exp_dir/data/truecase_model.$src \
  | python3 $bpe_scripts/apply_bpe.py \
      --codes $exp_dir/data/bpe.model \
      --vocabulary $exp_dir/data/vocab.$src \
      --vocabulary-threshold 10 \
  | $marian_bin/marian-decoder \
      --models $MODEL \
      --vocabs $working_dir/vocab.$src.yml $working_dir/vocab.$tgt.yml \
      --beam-size 12 \
      --normalize 1 \
      --word-penalty 0 \
      --mini-batch 16 \
      --maxi-batch 100 \
      --maxi-batch-sort src \
      "$@" \
  | perl -pe 's/@@ //g' \
  | $moses_scripts/recaser/detruecase.perl \
  | $moses_scripts/tokenizer/detokenizer.perl -q -l $tgt
