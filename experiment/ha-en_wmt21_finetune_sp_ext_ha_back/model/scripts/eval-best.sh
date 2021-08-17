#!/bin/bash

gpus=$1
shift

for model in ${@}; do
    echo $model
    echo "Evaluating best translation model in $model" 1>&2

    source $model/scripts/vars

   test -s $model/out/dev.out || cat $data_dir/raw/dev.tag.$src \
       | $marian_bin/marian-decoder -c $model/model/model.npz.best-bleu-detok.npz.decoder.yml -d $gpus --quiet > $model/out/dev.out

    $moses_scripts/generic/multi-bleu-detok.perl -lc $data_dir/raw/dev.raw.$tgt < $model/out/dev.out > $model/bleu/dev.bleu
    bleu_dev=$(cat $model/bleu/dev.bleu | sed -r 's/BLEU = ([0-9.]+),.*/\1/')

    # test -s $model/out/test.out || cat $data_dir/bpe/test.raw.$src \
    #     | $marian_bin/marian-decoder -c $model/model/model.npz.best-perplexity.npz.decoder.yml -d $gpus --quiet  > $model/out/test.out

    # $moses_scripts/generic/multi-bleu-detok.perl -lc $data_dir/raw/test.raw.$tgt < $model/out/test.out > $model/bleu/test.bleu
    # bleu_test=$(cat $model/bleu/test.bleu | sed -r 's/BLEU = ([0-9.]+),.*/\1/')

    # echo "dev= $bleu_dev test= $bleu_test"
done
