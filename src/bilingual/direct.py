####### Configuration begin #######

BASE = '/home/yunpengjiao/yunpengjiao/mscproject/'
BASE_EXP = '/home/yunpengjiao/yunpengjiao/mscproject/experiment/'
BASE_LOG = '/home/yunpengjiao/yunpengjiao/mscproject/log/'

PRNT_DIR = 'de-en_wmt21_pretrain_sp_ext_de'
CHLD_DIR = 'ha-en_wmt21_finetune_sp_ha.iter2_emb'

####### Configuration end #######

import numpy as np

prnt_model_path = f'{BASE_EXP}/{PRNT_DIR}/model/model/model.npz.best-translation.npz'
chld_model_path__last = f'{BASE_EXP}/{CHLD_DIR}/model/model/model.npz.best-bleu-detok.npz'
chld_model_path__next = f'{BASE_EXP}/{CHLD_DIR}/model/model/model.npz.best-bleu-detok.iter3.npz'

prnt_model = np.load(prnt_model_path)
chld_model_last = np.load(chld_model_path__last)

chld_model_next = {}
for k,v in prnt_model.items():
    chld_model_next[k] = v

chld_model_next['encoder_Wemb'] = chld_model_last['encoder_Wemb']
chld_model_next['decoder_Wemb'] = chld_model_last['decoder_Wemb']

np.savez(chld_model_path__next, **chld_model_next)