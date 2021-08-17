de = []
with open('./de.ha-en.raw.tok', 'r') as f:
    for l in f.readlines():
        de.append(l.replace('\n', ''))

ha = []
with open('./dev.raw.ha.tok', 'r') as f:
    for l in f.readlines():
        ha.append(l.replace('\n', ''))

with open('./de-ha.tok', 'w') as f:
    for d,h in zip(de,ha):
        f.write(f'{d} ||| {h}\n')