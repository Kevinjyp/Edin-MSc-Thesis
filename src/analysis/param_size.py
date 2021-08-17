EXP_DIR = '/home/yunpengjiao/yunpengjiao/mscproject/experiment/'

SUFFIX = '/model/model/model.npz'

METHODS = [
    'vocab_2k',
    'vocab_4k',
    'vocab_8k',
    'vocab_16k',
]

import numpy as np
import pdb

def tuple_mult(tup):
    ret = 1
    for t in tup:
        ret = ret * t
    return ret

for m in METHODS:

    model = np.load(f'{EXP_DIR}/ha-en_wmt21_sp.split_src_tgt.{m}/{SUFFIX}')

    num_params = 0
    for k,v in model.items():
        num_params += tuple_mult(v.shape)

    print(m, num_params)
    # pdb.set_trace()

# baseline
model = np.load(f'{EXP_DIR}/ha-en_wmt21_sp.split_src_tgt/{SUFFIX}')

num_params = 0
for k,v in model.items():
    num_params += tuple_mult(v.shape)

print('baseline', num_params)