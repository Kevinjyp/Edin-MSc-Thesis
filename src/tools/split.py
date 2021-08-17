de, en = [], []

FILE = 'europarl-v10.de-en.tsv'

with open(f'/fs/magni0/yunpengjiao/mscproject/data/para/de-en/{FILE}', 'r') as f:
    lines = f.readlines()
    for l in lines:
        de.append(l.split('\t')[0].replace('\n', ''))
        en.append(l.split('\t')[1].replace('\n', ''))
#import pdb;pdb.set_trace()
with open(f'/fs/magni0/yunpengjiao/mscproject/data/para/de-en/europarl-v10.de', 'w') as f:
    f.writelines("%s\n" % l for l in de)
    
with open(f'/fs/magni0/yunpengjiao/mscproject/data/para/de-en/europarl-v10.en', 'w') as f:
    f.writelines("%s\n" % l for l in en)
