de = []
with open('./de.ha-en.raw.tok', 'r') as f:
    for l in f.readlines():
        de.append(l.replace('\n', '').split(' '))

ha = []
with open('./dev.raw.ha.tok', 'r') as f:
    for l in f.readlines():
        ha.append(l.replace('\n', '').split(' '))

def format(array):
    ret = []
    for a in array:
        ret.append((int(a.split('-')[0]), int(a.split('-')[1])))
    return ret

align = []
with open('./de-ha.align', 'r') as f:
    for l in f.readlines():
        align.append(format(l.replace('\n', '').split(' ')))
try:
    align_tok = []
    for dd, hh, aa in zip(de,ha,align):
        for a in aa:
            d_idx, h_idx = a[0], a[1]
            align_tok.append('{}\t{}\n'.format(hh[h_idx], dd[d_idx]))
except:
    import pdb
    pdb.set_trace()

print(len(align_tok))

align_tok = list(tuple(align_tok))

print(len(align_tok))

with open('./de-ha.ft.align', 'w') as f:
    for a in align_tok:
        f.write(a.lower())