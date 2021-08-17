#!/bin/bash

gpus="$devices" 
model=$working_dir

echo $model
echo "Evaluating best translation model in $model" 1>&2

source $model/scripts/vars

cat /home/yunpengjiao/yunpengjiao/mscproject/experiment/ha-en_wmt21_finetune_sp_ext_ha/data/raw/dev.raw.en | $marian_bin/marian-decoder -c $model/model/model.npz.best-translation.npz.decoder.yml -d $gpus --quiet > /home/yunpengjiao/yunpengjiao/mscproject/monolingual/de-ha/de.ha-en.raw
