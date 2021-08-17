####### Configuration begin #######

# METHOD = 'rnd'
# METHOD = 'match-freq'
# METHOD = 'match-rnd'
# METHOD = 'levenshtein'

METHODS = ['rnd', 'match-freq', 'match-rnd', 'levenshtein']

BASE = '/home/yunpengjiao/yunpengjiao/mscproject/'
BASE_EXP = '/home/yunpengjiao/yunpengjiao/mscproject/experiment/'
BASE_LOG = '/home/yunpengjiao/yunpengjiao/mscproject/log/'

EXP_NAME = 'de-en_wmt21_pretrain_sp_de.split_src_tgt.vocab_upsample'

PARENT_VOCAB = f'{EXP_NAME}/data/spm/vocab.de.vocab'
CHILD_VOCAB = f'{EXP_NAME}/data/spm/vocab.ha.vocab'

PRNT_MODEL_PATH = f'{BASE_EXP}/{EXP_NAME}/model/model/model.npz.best-translation.npz'

STREAM = False

####### Configuration end #######

######### import begin ##########

import sentencepiece_model_pb2 as model
import sentencepiece as spm
import pdb
import random
import logging
import time
import os
import numpy as np
import sys
from concurrent.futures import ThreadPoolExecutor
import Levenshtein

sys.path.append(f'{BASE}/bilingual/')
import log
logger = log.get_logger(LOG_NAME=f'{BASE_LOG}/transfer', STREAM=STREAM)

########### import end ##########

class Vocab:
    def __init__(self, file_path: str, logger=logger):
        self.logger = logger
        self.path = file_path
        self.name = file_path.split('/')[-1]
        self.token_list = []
        self.token2idx, self.idx2token = {}, {}
        self.token_match = {}

        self.load(file_path)
        return

    def vocab_match(self, vocab):
        cnt_match = 0
        for token in self.token_list:
            if token in vocab.token2idx:
                # write new pose index
                self.token_match[token] = vocab.token2idx[token]
                cnt_match += 1
            else:
                # unmatch write -1
                self.token_match[token] = -1
        self.logger.info(f'{self.name} match {cnt_match} tokens with {vocab.name}.')
        return

    def load(self, path=None):
        self.num_tokens = 0
        with open(path, 'r', errors='replace') as f:
            lines = f.readlines()
            for idx,l in enumerate(lines):
                token = l.split('\t')[0]
                # pdb.set_trace()
                self.token_list.append(token)
                self.token2idx[token] = idx
                self.idx2token[idx] = token
                self.num_tokens += 1
        self.logger.info(f'read file {self.path} including {self.num_tokens} tokens')
        return


class TokenMapper:
    def __init__(self, prnt_vocab, chld_vocab, logger=logger):
        self.logger = logger
        self.prnt_vocab = prnt_vocab
        self.chld_vocab = chld_vocab
        self.mapping_methods = (
            'rnd',
            'match-freq',
            'match-rnd',
            'levenshtein',
        )
    
    def do(self, method='random'):
        if method not in self.mapping_methods:
            self.logger.error(f'{method} not match.')
            return
        elif method == 'rnd':
            self.map_random()
        elif method == 'match-freq':
            self.map_match_freq()
        elif method == 'match-rnd':
            self.map_match_rnd()
        elif method == 'levenshtein':
            self.map_levenshtein()
        else:
            self.logger.error(f'{method} not match.')
            return

        self.check_index()
        for i, (token, pos) in enumerate(self.chld_vocab.token_match.items()):
            self.logger.info(f'{i}, {token}, {pos}')


    def check_index(self):
        index_range = {}
        for i in range(self.chld_vocab.num_tokens):
            index_range[i] = 0
        
        for token, idx in self.chld_vocab.token_match.items():
            if idx not in index_range:
                self.logger.error(f'Token {token} index {idx} valid for {self.chld_vocab.name}')
            if index_range[idx] != 0:
                self.logger.error(f'Token {token} index {idx} duplicate for {self.chld_vocab.name}')
            index_range[idx] = 1

        index_range = {}
        for i in range(self.prnt_vocab.num_tokens):
            index_range[i] = 0

        for token, idx in self.prnt_vocab.token_match.items():
            if idx not in index_range:
                self.logger.error(f'Token {token} index {idx} valid for {self.prnt_vocab.name}')
            if index_range[idx] != 0:
                self.logger.error(f'Token {token} index {idx} duplicate for {self.prnt_vocab.name}')
            index_range[idx] = 1
        self.logger.info('Index check pass!')
        return


    def map_random(self):
        self.logger.info('start execute random mapping')
        
        t_list = list(range(self.chld_vocab.num_tokens))
        random.shuffle(t_list)

        for idx, (token, _) in enumerate(self.chld_vocab.token_match.items()):
            self.chld_vocab.token_match[token] = t_list[idx]
        
        for idx, (token, _) in enumerate(self.prnt_vocab.token_match.items()):
            self.prnt_vocab.token_match[token] = t_list[idx]
        
        self.logger.info('random mapping end')
        return

    def map_match_freq(self):
        self.logger.info('start execute match-frequency mapping')
        for token, pos_new in self.chld_vocab.token_match.items():
            if pos_new == -1:
                chld_token_idx = self.chld_vocab.token2idx[token]
                
                for prnt_token, pos in self.prnt_vocab.token_match.items():
                    if pos == -1:
                        prnt_token_idx = self.prnt_vocab.token2idx[prnt_token]

                        self.prnt_vocab.token_match[prnt_token] = chld_token_idx
                        self.chld_vocab.token_match[token] = prnt_token_idx
                        break
        self.logger.info('match-frequency mapping end')
        return

    def map_match_rnd(self):
        self.logger.info('start execute match-random mapping')

        prnt_pos_list = []
        for token, pos_new in self.prnt_vocab.token_match.items():
            if pos_new == -1:
                prnt_pos_list.append(self.prnt_vocab.token2idx[token])
        random.shuffle(prnt_pos_list)

        for token, pos_new in self.chld_vocab.token_match.items():
            if pos_new == -1:
                chld_token_idx = self.chld_vocab.token2idx[token]
                
                prnt_token_idx = prnt_pos_list.pop()
                prnt_token = self.prnt_vocab.idx2token[prnt_token_idx]

                self.prnt_vocab.token_match[prnt_token] = chld_token_idx
                self.chld_vocab.token_match[token] = prnt_token_idx

        self.logger.info('match-random mapping end')
        return

    def map_levenshtein(self):
        self.logger.info('start execute levenshtein mapping')

        prnt_pos_list = []
        for token, pos_new in self.prnt_vocab.token_match.items():
            if pos_new == -1:
                prnt_pos_list.append(self.prnt_vocab.token2idx[token])
        random.shuffle(prnt_pos_list)

        for token, pos_new in self.chld_vocab.token_match.items():
            if pos_new == -1:
                chld_token_idx = self.chld_vocab.token2idx[token]

                min_dist = 99999
                min_idx = -1

                for prnt_token_idx in prnt_pos_list:

                    prnt_token = self.prnt_vocab.idx2token[prnt_token_idx]

                    dist = Levenshtein.distance(token, prnt_token)
                    if min_dist > dist:
                        min_dist = dist
                        min_idx = prnt_token_idx
                
                prnt_token = self.prnt_vocab.idx2token[min_idx]
                self.prnt_vocab.token_match[prnt_token] = chld_token_idx
                self.chld_vocab.token_match[token] = min_idx
                prnt_pos_list.remove(min_idx)

        self.logger.info('match-random mapping end')
        return


class EmbeddingTransfer:
    def __init__(self, prnt_vocab, chld_vocab, prnt_model_path, chld_model_path, logger=logger):
        self.logger = logger
        self.prnt_vocab = prnt_vocab
        self.chld_vocab = chld_vocab
        self.prnt_model_path = prnt_model_path
        self.chld_model_path = chld_model_path

        # load model
        self.prnt_model = np.load(prnt_model_path)
        self.logger.info(f'EmbeddingTransfer load parent model from {prnt_model_path} ')

    def trans_statistic(self):
        def trans(pos_new):
            return old_encoder_embed[pos_new], old_decoder_embed[pos_new]

        old_encoder_embed = self.prnt_model['encoder_Wemb'].copy()
        old_decoder_embed = self.prnt_model['decoder_Wemb'].copy()

        new_encoder_embed = self.prnt_model['encoder_Wemb'].copy()
        new_decoder_embed = self.prnt_model['decoder_Wemb'].copy()

        self.logger.info(f'Start manipulating the model.')

        executor = ThreadPoolExecutor(256)
        
        pos_news = []
        for idx, (_, pos_new) in enumerate(self.chld_vocab.token_match.items()):
            pos_news.append(pos_new) 
        
        for idx, result in enumerate(executor.map(trans, pos_news)):
            new_encoder_embed[idx] = result[0]
            new_decoder_embed[idx] = result[1]

        a = np.array_equal(new_encoder_embed, self.prnt_model['encoder_Wemb'])
        b = np.array_equal(new_decoder_embed, self.prnt_model['encoder_Wemb'])
        
        self.logger.info(f'encoder_Wemb: {a}')
        self.logger.info(f'decoder_Wemb: {b}')

        chld_model = {}
        for k,v in list(self.prnt_model.items()):
            chld_model[k] = v
        
        chld_model['encoder_Wemb'] = new_encoder_embed
        chld_model['decoder_Wemb'] = new_decoder_embed

        self.logger.info(f'Saving model file.')

        np.savez(self.chld_model_path, **chld_model)


if __name__ == '__main__':
    for method in METHODS:

        de_vocab = Vocab(f'{BASE_EXP}/{PARENT_VOCAB}')
        ha_vocab = Vocab(f'{BASE_EXP}/{CHILD_VOCAB}')

        de_vocab.vocab_match(ha_vocab)
        ha_vocab.vocab_match(de_vocab)

        m = TokenMapper(de_vocab, ha_vocab)  
        m.do(method)

        CHLD_MDOEL_PATH = f'{BASE_EXP}/{EXP_NAME}/model/model/model.npz.best-translation.trans-{method}.npz'

        t = EmbeddingTransfer(de_vocab, ha_vocab, PRNT_MODEL_PATH, CHLD_MDOEL_PATH)
        t.trans_statistic()
        
    pdb.set_trace() 
