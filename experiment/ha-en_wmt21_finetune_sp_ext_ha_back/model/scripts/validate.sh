#!/bin/bash

. `dirname $0`/vars

cat $1 | tee $data_dir/bpe/dev.bpe.in > $working_dir/out/dev.out

$moses_scripts/generic/multi-bleu-detok.perl -lc $data_dir/raw/dev.raw.$tgt < $working_dir/out/dev.out 2>/dev/null | sed -r 's/BLEU = ([0-9.]+),.*/\1/'
