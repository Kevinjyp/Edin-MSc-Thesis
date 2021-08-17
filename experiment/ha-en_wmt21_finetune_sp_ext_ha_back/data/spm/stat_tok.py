import numpy as np

with open(f'./vocab.ha-en.vocab', 'r') as f:
    len_tokens = []
    tokens = []
    for l in f.readlines():
        token = l.split('\t')[0].replace('▁', '') 
        if token in ('<unk>','<s>','</s>'):
            continue
        tokens.append(token)
        len_tokens.append(len(token))
    # import pdb
    # pdb.set_trace()
    print(f'mean:{np.mean(len_tokens)}\tmin{np.min(len_tokens)}\tmax{np.max(len_tokens)}\tstd{np.std(len_tokens)}')