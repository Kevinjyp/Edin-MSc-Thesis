#!/bin/bash

gpus=$1
shift

for model in $PWD; do
    echo $model
    echo "Evaluating best translation model in $model" 1>&2

    source $model/scripts/vars

   cat $model/train.bpe.trunc.$src \
       | $marian_bin/marian-decoder -c $model/model.npz.best-translation.npz.decoder.yml -d $gpus --quiet \
        > $model/train.out

done
