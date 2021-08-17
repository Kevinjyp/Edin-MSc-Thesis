#!/bin/sh

mydir=`dirname $0`
. $mydir/vars

WILKES=cs-hadd1@login-gpu.hpc.cam.ac.uk
DEST=/home/cs-hadd1/gourmet-y1-bg-en/


DEST=$DEST/run$expt
echo "Uploading to $DEST"
ssh $WILKES mkdir -p $DEST

for lang in $src $tgt; do 
  scp $data_dir/train-${train}.bpe.$lang $WILKES:$DEST
  scp $backtrans_dir/news.bpe.$tgt-$src.$lang $WILKES:$DEST
  for segment in test dev; do
    scp $data_dir/$segment.bpe.$lang $WILKES:$DEST
    scp $data_dir/$segment.raw.$lang $WILKES:$DEST/$segment.$lang
  done
done

scp -r $mydir $WILKES:$DEST
ssh $WILKES mkdir -p $DEST/logs
ssh $WILKES perl -pi -e "s/__EXPT__/$expt/g" $DEST/scripts/submit


