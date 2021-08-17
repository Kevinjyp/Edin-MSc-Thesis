import pdb

BASE = '/home/yunpengjiao/yunpengjiao/mscproject/experiment/ha-en_wmt21_finetune_sp_ext_ha_back/data/spm'
FILE_1 = 'vocab.ha-en.en'
FILE_2 = 'vocab.ha-en.ha'

s1, s2 = 0, 0

def union(vocab1, vocab2):
    v1 = set(vocab1.keys())
    v2 = set(vocab2.keys())
    return v1.union(v2)

def load_vocab(file):
    vocab = {}
    with open(f"{file}") as fin:
        for line in fin:
            word, number = line.strip().split("\t")
            vocab[word] = int(number)
    return vocab

def dump_vocab(v1_only, v2_only, joint):
    with open(f'{BASE}/{FILE_1}.split', 'w') as f:
        f.writelines([v+'\n' for v in v1_only])
        f.writelines([v+'\n' for v in joint])

    with open(f'{BASE}/{FILE_2}.split', 'w') as f:
        f.writelines([v+'\n' for v in v2_only])
        f.writelines([v+'\n' for v in joint])

def ratio(vocab1, vocab2, word, gamma=0.05):
    if word not in vocab1 or word not in vocab2:
        return False
    c1 = vocab1[word]
    c2 = vocab2[word]
    return (gamma) <= (s2 / s1) * (c1 / c2) <= (1 / gamma)

def split(vocab1, vocab2):
    v1 = set()
    v2 = set()
    joint = set()
    for word in union(vocab1, vocab2):
        if ratio(vocab1, vocab2, word):
            joint.add(word)
        elif vocab1.get(word, 0) > vocab2.get(word, 0):
            v1.add(word)
        else:
            v2.add(word)
    return v1, v2, joint


if __name__ == '__main__':
    v1 = load_vocab(f'{BASE}/{FILE_1}')
    v2 = load_vocab(f'{BASE}/{FILE_2}')

    s1 = sum(list(v1.values()))
    s2 = sum(list(v2.values()))

    v1_only, v2_only, joint = split(v1, v2)
    dump_vocab(v1_only, v2_only, joint)
