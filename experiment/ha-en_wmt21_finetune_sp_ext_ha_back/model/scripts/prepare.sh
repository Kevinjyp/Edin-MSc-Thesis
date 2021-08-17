#!/bin/sh

. `dirname $0`/vars

# mkdir -p logs

# if [ ! -e $working_dir/train.bpe.$src ]; then
#   for lang in $src $tgt; do
#     cp $data_dir/dev.bpe.$lang .
#     cp $data_dir/test.bpe.$lang .
#     cp $data_dir/dev.raw.$lang dev.$lang
#     cp $data_dir/test.raw.$lang test.$lang
#     cat  $data_dir/train-$train.bpe.$lang >> $working_dir/train.bpe.$lang
#   done
# fi

subdirs="out model bleu logs"

for subdir in $subdirs; do
    mkdir -p $working_dir/$subdir
    echo $working_dir/$subdir
done

if [ ! -e $working_dir/vocab.$src.yml ]; then
    cat $data_dir/bpe/vocab.$src | $marian_bin/marian-vocab > $data_dir/bpe/vocab.$src.yml
fi

if [ ! -e $working_dir/vocab.$tgt.yml ]; then
    cat $data_dir/bpe/vocab.$tgt | $marian_bin/marian-vocab > $data_dir/bpe/vocab.$tgt.yml
fi