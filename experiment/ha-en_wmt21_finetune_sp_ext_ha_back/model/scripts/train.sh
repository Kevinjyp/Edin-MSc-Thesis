#!/bin/bash


. `dirname $0`/vars

# preperation
bash $working_dir/scripts/prepare.sh

# train
$marian_bin/marian \
    -d $devices \
    --pretrained-model  /home/yunpengjiao/yunpengjiao/mscproject/experiment/de-en_wmt21_pretrain_sp_ext_de/model/model/model.npz.best-translation.npz \
    -c $working_dir/scripts/config-trans.yml \
    -m $working_dir/model/model.npz \
    --mini-batch-fit -w 5000  --mini-batch 1000 --maxi-batch 1000 \
    --train-sets $data_dir/raw/train-all.raw.ha-en.$src $data_dir/raw/train-all.raw.ha-en.$tgt \
    --vocabs $data_dir/spm/vocab.ha-en.spm $data_dir/spm/vocab.ha-en.spm \
    --sentencepiece-max-lines 0 \
    --valid-script-path $working_dir/scripts/validate.sh \
    --valid-sets $data_dir/raw/dev.tag.$src $data_dir/raw/dev.raw.$tgt \
    --valid-log $working_dir/logs/valid.log \
    --log $working_dir/logs/train.log


test -e $working_dir/model/model.npz || exit 1

# eval best
bash $working_dir/scripts/eval-best.sh "$devices" $working_dir > $working_dir/logs/eval.log

# eval ensemble
mv $working_dir/scripts/model.npz.ensemble.npz.decoder.yml $working_dir/model
