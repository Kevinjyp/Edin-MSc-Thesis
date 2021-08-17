#!/bin/bash

gpus=$1
shift

for model in ${@}; do
    echo $model
    echo "Evaluating best translation model in $model" 1>&2

    source $model/scripts/vars

   test -s $model/out/dev.ensemble.out || cat $data_dir/bpe/dev.bpe.$src \
       | $marian_bin/marian-decoder -c $model/model/model.npz.ensemble.npz.decoder.yml -d $gpus \
       | perl -pe 's/@@ //g' \
       | $moses_scripts/recaser/detruecase.perl \
       | $moses_scripts/tokenizer/detokenizer.perl -q -l en > $model/out/dev.ensemble.out

    $moses_scripts/generic/multi-bleu-detok.perl -lc $data_dir/raw/dev.raw.$tgt < $model/out/dev.ensemble.out > $model/bleu/dev.ensemble.bleu
    bleu_dev=$(cat $model/bleu/dev.bleu | sed -r 's/BLEU = ([0-9.]+),.*/\1/')

    # test -s $model/out/test.ensemble.out || cat $data_dir/bpe/test.bpe.$src \
    #     | $marian_bin/marian-decoder -c $model/model/model.npz.ensemble.npz.decoder.yml -d $gpus --quiet \
    #     | perl -pe 's/@@ //g' \
    #     | $moses_scripts/recaser/detruecase.perl \
    #     | $moses_scripts/tokenizer/detokenizer.perl -q -l en 2>/dev/null > $model/out/test.ensemble.out

    # $moses_scripts/generic/multi-bleu-detok.perl -lc $data_dir/raw/test.raw.$tgt < $model/out/test.ensemble.out > $model/bleu/test.ensemble.bleu
    # bleu_test=$(cat $model/bleu/test.bleu | sed -r 's/BLEU = ([0-9.]+),.*/\1/')

    echo "dev= $bleu_dev test= $bleu_test"
done
