import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# with open('/fs/magni0/yunpengjiao/mscproject/data/para/ha-en/khamenei.v1.ha-en.tsv', 'r') as f:
#     lines = f.readlines()
#     raw_scores = [l.split('\t')[-1] for l in lines]
#     scores = [float(s.replace('\n', '')) for s in raw_scores]
#     plt.hist(scores, bins=20)
#     plt.title('khamenei score histogram')
#     plt.xlabel('Score')
#     plt.ylabel('#Sentences')
#     plt.savefig('/fs/magni0/yunpengjiao/mscproject/data/para/ha-en/khamenei_hist.png')
#     # import pdb; pdb.set_trace()

with open('/fs/magni0/yunpengjiao/mscproject/data/para/ha-en/paracrawl8/paracrawl-release8.en-ha.bifixed.dedup.laser.filter-0.9', 'r') as f:
    lines = f.readlines()
    raw_scores = [l.split('\t')[0] for l in lines]
    scores = [float(s) for s in raw_scores]
    plt.hist(scores, bins=20)
    plt.title('paracrawl8 en-ha score histogram')
    plt.xlabel('Score')
    plt.ylabel('#Sentences')
    plt.savefig('/fs/magni0/yunpengjiao/mscproject/data/para/ha-en/paracrawl8_hist.png')
    # import pdb; pdb.set_trace()