EXP_DIR = '/home/yunpengjiao/yunpengjiao/mscproject/experiment/'

SUFFIX = '/model/logs/valid.log'

TITLES =[
    [
        'Joint src&tgt with Parallel data',
        'Joint src&tgt + back-trans data (parent)',
        'Joint src&tgt + back-trans data (child)',
    ], 
    [
        'Separate src&tgt with Parallel data',
        'Separate src&tgt + back-trans data (parent)',
        'Separate src&tgt + back-trans data (child)',
    ]
]

PREFIXS = [
    [
        'ha-en_wmt21_finetune_sp_ha',
        'ha-en_wmt21_finetune_sp_ext_ha',
        'ha-en_wmt21_finetune_sp_ext_ha_back',
    ],
    [
        'ha-en_wmt21_finetune_sp_ha.split_src_tgt',
        'ha-en_wmt21_finetune_sp_ext_ha.split_src_tgt',
        'ha-en_wmt21_finetune_sp_ext_ha_back.split_src_tgt',
    ]
]

BASELINES = [
    [
        'ha-en_wmt21_sp',
        'ha-en_wmt21_sp',
        'ha-en_wmt21_sp_back',
    ],
    [
        'ha-en_wmt21_sp.split_src_tgt',
        'ha-en_wmt21_sp.split_src_tgt',
        'ha-en_wmt21_sp_back.split_src_tgt',
    ]
]

# VOCAB_IDX = 1
# DATA_IDX = 1

# TITLE = TITLES[VOCAB_IDX][DATA_IDX]
# PREFIX = PREFIXS[VOCAB_IDX][DATA_IDX]
# BASELINE = BASELINES[VOCAB_IDX][DATA_IDX]


METHODS = ('trans_rnd', 'trans_match_freq', 'trans_match_rnd', 'trans_levenshtein')
LEGENDS = ('all rnd', 'spelling+freq', 'spelling+rnd', 'levenshtein')
KEYWORD = 'bleu-detok'

INPUT = '/home/yunpengjiao/yunpengjiao/mscproject/experiment/ha-en_wmt21_finetune_sp_ext_ha.trans_levenshtein/model/logs/valid.log'
OUTPUT = '/home/yunpengjiao/yunpengjiao/mscproject/picture'
TYPES = ['png', 'svg', 'pdf']

DOWN_RATIO = 4

MARKERS = ['o', 's', '^', 'D', '*', 'x']

FONT_SIZE = 15

import matplotlib
import matplotlib.pyplot as plt

def down_sample(list, scale=1):
    ret = []
    for idx, l in enumerate(list):
        if idx % scale == 0:
            ret.append(l)
    return ret

def runtime_stat(file, keyword):

    steps, bleus = [], []

    with open(file, 'r') as f:
        lines = f.readlines()
        for l in lines:
            if keyword in l:
                step = int(l.split('Up. ')[1].split(' :')[0])
                bleu = float(l.split('bleu-detok : ')[1].split(' :')[0])

                steps.append(step)
                bleus.append(bleu)

    max_bleu = max(bleus)
    max_step = steps[bleus.index(max_bleu)]

    print(file, keyword, max_step, max_bleu)
    # print(steps)
    # print(bleus)

    return steps, bleus

# runtime_stat(INPUT, KEYWORD)

def my_format(x, pos):
    if x == 0:
        return 0
    else:
        return str(int(x/1000)) + 'k'


def main():
    fig, axs = plt.subplots()
    axs.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(my_format)
    )

    # different mapping methods
    for idx,method in enumerate(METHODS):
        file = f'{EXP_DIR}/{PREFIX}.{method}/{SUFFIX}'
        steps, bleus = runtime_stat(file, KEYWORD)
        plt.plot(down_sample(steps, DOWN_RATIO), down_sample(bleus, DOWN_RATIO), label=LEGENDS[idx], markersize=10, marker=MARKERS[idx])

    # all freq
    file = f'{EXP_DIR}/{PREFIX}/{SUFFIX}'
    steps, bleus = runtime_stat(file, KEYWORD)
    plt.plot(down_sample(steps, DOWN_RATIO), down_sample(bleus, DOWN_RATIO), label='all freq', markersize=10, marker=MARKERS[len(METHODS)])

    # baseline
    file = f'{EXP_DIR}/{BASELINE}/{SUFFIX}'
    steps, bleus = runtime_stat(file, KEYWORD)
    plt.plot(down_sample(steps, DOWN_RATIO), down_sample(bleus, DOWN_RATIO), label='train-from-scratch', markersize=10, marker=MARKERS[len(METHODS)+1])

    plt.xticks(fontsize=FONT_SIZE-4)
    plt.yticks(fontsize=FONT_SIZE-4)
    plt.xlabel('Training Step', fontsize=FONT_SIZE)
    plt.ylabel('BLEU points (*100)', fontsize=FONT_SIZE)
    plt.title(f'{TITLE}', fontsize=FONT_SIZE)
    plt.legend(fontsize=FONT_SIZE)
    for t in TYPES:
        plt.savefig(f'{OUTPUT}/{t}/{PREFIX}.{t}')
    plt.clf()


for i in (0,1):
    for j in (0,1,2):

        TITLE = TITLES[i][j]
        PREFIX = PREFIXS[i][j]
        BASELINE = BASELINES[i][j]

        main()