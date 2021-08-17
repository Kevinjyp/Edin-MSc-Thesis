#!/bin/bash

spm_encode --model=../ha-en/ha-en.spm/vocab.ha-en.model --output_format=piece < ./dev.raw.ha > ./dev.raw.ha.tok


spm_encode --model=../de-en/de-en.spm/vocab.de-en.model --output_format=piece < ./de.ha-en.raw > ./de.ha-en.raw.tok