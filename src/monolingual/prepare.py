####### Configuration begin #######

BASE = '/home/yunpengjiao/yunpengjiao/mscproject/'
BASE_EXP = '/home/yunpengjiao/yunpengjiao/mscproject/experiment/'
BASE_LOG = '/home/yunpengjiao/yunpengjiao/mscproject/log/'

CHILD_SPM = 'ha-en_wmt21_finetune_sp_ha.split_src_tgt/data/spm/vocab.ha.vocab'
CHILD_FASTTEXT = 'monolingual/ha/data/corpus.ha.tok.emb.vec'

PARENT_DIR = 'de-en_wmt21_pretrain_sp_de.split_src_tgt'
MODEL_PATH = 'model/model/model.npz.best-translation.npz'
VOCAB_PATH = 'data/spm/vocab.de.vocab'

MUSE_DIR = '/home/yunpengjiao/yunpengjiao/repos/MUSE/dumped/debug'
DIR_NAME = [
    'gcpkl3bszb',
    'y916kscnmk',
]
MUSE_EXP_MAP = {
    'gcpkl3bszb': 'de-en_wmt21_pretrain_sp_de.split_src_tgt',
    'y916kscnmk': 'de-en_wmt21_pretrain_sp_ext_de.split_src_tgt',
}

VEC_PATH = 'vectors-ha.txt'

STREAM = False

####### Configuration end #######


######### import begin ##########

import pdb
import sys
import numpy as np

sys.path.append(f'{BASE}/bilingual/')
import log, transfer

logger = log.get_logger(LOG_NAME=f'{BASE_LOG}/MUSE', STREAM=STREAM)

########### import end ##########


class Vec:
    def __init__(self, logger=logger):
        self.logger = logger
        self.value = []
        self.token2idx, self.idx2token = {}, {}


    def load_from_array(self, tokens, values):
        # values: numpy array
        # tokens: [token]
        if values.shape[0] != len(tokens):
            logger.error(f'shape miss match: values {values.shape}, tokens {len(tokens)}')
            return
        for idx, value in enumerate(values):
            self.value.append(value)
            self.token2idx[tokens[idx]] = idx
            self.idx2token[idx] = tokens[idx]
            logger.info(f'load {tokens[idx]} with {idx}')
        pass
    

    def format(self, token, value):
        ret = f'{token}'
        for e in list(value):
            ret += " %.6f" % e
        return ret


    def dump(self, path):
        embed_dim = self.value[0].shape[-1]
        num_tokens = len(self.value)
        with open(path, 'w') as f:
            f.write(f'{num_tokens} {embed_dim}\n')
            for token in list(self.token2idx.keys()):
                value = self.value[self.token2idx[token]]
                f.write(self.format(token, value) + '\n')
                logger.info(f'[dump] {self.format(token, value)}')
        return


    def sort(self, spm_vocab):
        sorted_tokens = []
        for token, _ in spm_vocab.token2idx.items():
            sorted_tokens.append(token)

        sorted_values = []
        for sort_tok in sorted_tokens:
            sorted_values.append(
                self.value[self.token2idx[sort_tok]]
        )
        return sorted_values


    def load_from_file(self, path):
        self.path = path
        with open(self.path, 'r') as f:
            lines = f.readlines()

            logger.info(f'load {self.path} file.')

            for idx, l in enumerate(lines[1:]):
                l = l.replace(' \n', '')
                l = l.replace('\n', '')

                token = l.split(' ')[0]
                try:
                    # 一个巨大的坑
                    # FAST_TEXT生成的corpus.ha-en.tok.emb.vec每行最后有一个空格
                    # 但是MUSE生成的text，最后一行结尾没有
                    # self.value.append(np.array([float(t) for t in l.split(' ')[1:-1]]))
                    self.value.append(np.array([float(t) for t in l.split(' ')[1:]]))
                except:
                    import pdb; pdb.set_trace()
                self.token2idx[token] = idx
                self.idx2token[idx] = token
            
            logger.info(f'all {len(list(self.token2idx.keys()))} tokens.')
  
            self.value = np.array(self.value)
            self.mean = np.mean(self.value, axis=0)
            self.cov = np.cov(self.value.T)

            logger.info(f'shape: {self.value.shape}')
            logger.info(f'mean: {self.mean.shape}')
            logger.info(f'std: {self.cov.shape}')
    
    def match(self, spm_vocab):
        vec_more = []
        spm_more = []

        # match tokens
        cnt_match = 0
        for token, _ in spm_vocab.token2idx.items():
            if token in self.token2idx:
                cnt_match += 1
                logger.debug(f'[MATCH] {token}')
            else:
                spm_more.append(token)
        logger.info(f'match {cnt_match} tokens with {spm_vocab.name} spm file.')
        logger.info(f'spm_more list size {len(spm_more)}.')

        for token, _ in self.token2idx.items():
            if token not in spm_vocab.token2idx:
                vec_more.append(token)
        logger.info(f'vec_more list size {len(vec_more)}.')

        # reuse vec tokens 
        cnt_reuse = 0
        for vec_tok in vec_more:
            spm_tok = spm_more.pop()
            self.token2idx[spm_tok] = self.token2idx[vec_tok]
            del self.token2idx[vec_tok]
            cnt_reuse += 1
            logger.debug(f'[REUSE] {vec_tok} -> {spm_tok}')
        logger.info(f'reuse {cnt_reuse} tokens.')

        # remaining spm tokens
        cnt_remain = 0
        for tok in spm_more:
            self.token2idx[tok] = -1
            cnt_remain += 1
            logger.debug(f'[REMAIN] {tok}!')
        logger.info(f'remaining {cnt_remain} tokens need init.')

        self.gaussian_init(cnt_remain)

    def gaussian_init(self, num):
        logger.info(f'begin gaussian init for {num} tokens')
        
        rnd_vec = np.random.multivariate_normal(self.mean, self.cov, num)
        
        idx_begin = self.value.shape[0]
        logger.info(f'random index begin at {idx_begin}.')

        self.value = np.vstack((self.value, rnd_vec))

        for token, idx in self.token2idx.items():
            if idx == -1:
                self.token2idx[token] = idx_begin
                self.idx2token[idx_begin] = token
                idx_begin += 1

        logger.info(f'random index end at {idx_begin}(not used).')


if __name__ == '__main__':
    # # child vocab 
    # ha_ft = Vec()
    # ha_ft.load_from_file(f'{BASE}/{CHILD_FASTTEXT}')

    # ha_spm = transfer.Vocab(f'{BASE_EXP}/{CHILD_SPM}', logger=logger)

    # ha_ft.match(ha_spm)
    # ha_ft.dump(f'{BASE}/{CHILD_FASTTEXT}.full')



    # parent vocab
    # model = np.load(f'{BASE_EXP}/{PARENT_DIR}/{MODEL_PATH}')
    # encoder_emb = model['encoder_Wemb']
    # logger.info(f'encoder_emb shape {encoder_emb.shape}')
    
    # tokens = []
    # with open(f'{BASE_EXP}/{PARENT_DIR}/{VOCAB_PATH}', 'r') as f:
    #     for l in f.readlines():
    #         token = l.split('\t')[0]
    #         tokens.append(token)
    #         logger.info(f'read token {token}')
    
    # de_ft = Vec()
    # de_ft.load_from_array(tokens, encoder_emb)
    # de_ft.dump(f'{BASE}/monolingual/de-en/output/{PARENT_DIR}.vec')

    # overwrite model file 
    for name in DIR_NAME:
        ha_ft = Vec()
        ha_ft.load_from_file(f'{MUSE_DIR}/{name}/{VEC_PATH}')
        ha_spm = transfer.Vocab(f'/home/yunpengjiao/yunpengjiao/mscproject/experiment/ha-en_wmt21_finetune_sp_ha.split_src_tgt/data/spm/vocab.ha.vocab', logger=logger)

        encoder_emb = ha_ft.sort(ha_spm)

        model = np.load(f'{BASE_EXP}/{MUSE_EXP_MAP[name]}/{MODEL_PATH}')

        model_new = {}
        for k,v in model.items():
            model_new[k] = v

        model_new['encoder_Wemb'] = np.array(encoder_emb)
        # print(model['encoder_Wemb'] == model_new['encoder_Wemb'])

        np.savez(f'{BASE_EXP}/{MUSE_EXP_MAP[name]}/{MODEL_PATH}.muse', **model_new)

        print(f'{BASE_EXP}/{MUSE_EXP_MAP[name]}/{MODEL_PATH}.muse')

        # pdb.set_trace()