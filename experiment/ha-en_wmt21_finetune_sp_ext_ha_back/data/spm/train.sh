#! /bin/bash
~/yunpengjiao/repos/marian-dev/build-elli/spm_train --input=train.en,train.ha --vocab_size=32000 --character_coverage=1.0 --model_type=bpe --model_prefix=vocab.ha-en