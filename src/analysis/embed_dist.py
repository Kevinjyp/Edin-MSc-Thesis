####### Configuration begin #######

BASE = '/home/yunpengjiao/yunpengjiao/mscproject/'
BASE_EXP = '/home/yunpengjiao/yunpengjiao/mscproject/experiment/'
BASE_LOG = '/home/yunpengjiao/yunpengjiao/mscproject/log/'
MODEL_PATH = 'model/model/model.npz'

PARENT_DIRS = [
    'de-en_wmt21_pretrain_sp_de',
    'de-en_wmt21_pretrain_sp_ext_de',
    'de-en_wmt21_pretrain_sp_ext_de',
]

JOINT_SEPARATE = [
    '',
    '.split_src_tgt',
]

CHILD_DIRS = [
    'ha-en_wmt21_finetune_sp_ha',
    'ha-en_wmt21_finetune_sp_ext_ha',
    'ha-en_wmt21_finetune_sp_ext_ha_back',
]

PARENT_MODEL_SUFFIXS = [
    '',
    '.muse.npz',
    '.trans-levenshtein.npz',
    '.trans-match-freq.npz',
    '.trans-match-rnd.npz',
    '.trans-rnd.npz',
]

CHILD_DIR_SUFFIXS = [
    '',
    '.muse',
    '.trans_levenshtein',
    '.trans_match_freq',
    '.trans_match_rnd',
    '.trans_rnd',
]

####### Configuration end #######

######### import begin ##########

import pdb
import sys
import numpy as np

sys.path.append(f'{BASE}/bilingual/')
import log

logger = log.get_logger(LOG_NAME=f'{BASE_LOG}/embedding_distance', STREAM=False)

########### import end ##########

for joint_separate in JOINT_SEPARATE:
    for parent_dir, child_dir in zip(PARENT_DIRS, CHILD_DIRS):
        for child_dir_suffix, parent_model_suffix in zip(CHILD_DIR_SUFFIXS, PARENT_MODEL_SUFFIXS):
            dists = []

            parent_model_path = f'{BASE_EXP}/{parent_dir}{joint_separate}/{MODEL_PATH}.best-translation.npz{parent_model_suffix}'
            child_model_path = f'{BASE_EXP}/{child_dir}{joint_separate}{child_dir_suffix}/{MODEL_PATH}.best-bleu-detok.npz'

            logger.info(f'parent_model_path: {parent_model_path}')
            parent_model = np.load(parent_model_path)
            parent_model_emb = parent_model['encoder_Wemb']

            logger.info(f'child_model_path: {child_model_path}')
            child_model = np.load(child_model_path)
            child_model_emb = child_model['encoder_Wemb']

            for prnt_emb, chld_emb in zip(parent_model_emb, child_model_emb):
                dists.append(np.linalg.norm(chld_emb - prnt_emb))
            
            dists_mean = np.mean(dists)
            dists_std = np.std(dists)

            logger.info(f'dists_mean: {dists_mean}')
            logger.info(f'dists_std: {dists_std}')

            # pdb.set_trace()