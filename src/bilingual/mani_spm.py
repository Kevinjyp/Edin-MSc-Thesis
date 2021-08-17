import sentencepiece_model_pb2 as model
import sentencepiece as spm
import pdb

# https://github.com/google/sentencepiece/issues/121
# https://pypi.org/project/sentencepiece/

m = model.ModelProto()
m.ParseFromString(open('/home/yunpengjiao/yunpengjiao/mscproject/experiment/de-en_wmt21_pretrain_sp/data/spm/vocab.de-en.model', 'rb').read())

# sp = spm.SentencePieceProcessor()
# sp.LoadFromSerializedProto(m.SerializedToString())
# sp.LoadFromSerializedProto(open('./vocab.de-en.model', 'rb').read())

print(4, m.pieces[4])
print(5, m.pieces[5])

m.pieces[4].piece, m.pieces[5].piece = m.pieces[5].piece, m.pieces[4].piece

m.pieces[4].score, m.pieces[5].score = m.pieces[5].score, m.pieces[4].score

print(4, m.pieces[4])
print(5, m.pieces[5])

pdb.set_trace()