from models.biobert_embedding.embedding import BiobertEmbedding as BIO_BERT
import json
import torch
import os
import numpy as np
source_file = '../../../data/abstract/seq2seq/sentence_labeled.json'
labeled_folder = '../../../data/abstract/seq2seq/embedding/'

from utils.data_loader import *

EMBEDDING = BiobertEmbedding()
PATH = os.path.dirname(os.path.realpath(__file__))




def create_embedding(js):
    for ques_id, ques in js.items():
        count = -1
        for sen, label in ques['label'].items():
            count += 1
            create = False
            if os.path.isfile(labeled_folder + str(ques_id) + '_' + str(count) + '.npy') or os.path.isfile(
                    labeled_folder + str(ques_id) + '_' + str(count) + '.txt'):
                pass
            else:
                try:
                    vector = BIO_BERT().sentence_vector(sen).tolist()
                    print('saving {} file'.format(str(ques_id) + '_' + str(count) + '.npy'))
                    np.save(vector, open(labeled_folder + str(ques_id) + '_' + str(count) + '.npy','wb'))
                except:
                    vector = sen
                    print('except sentence: {}'.format(sen))
                    print('saving {} file'.format(str(ques_id) + '_' + str(count) + '.txt'))
                    with open(labeled_folder + str(ques_id) + '_' + str(count) + '.txt', 'w+') as f:
                        f.write(vector)
                        f.close()


if __name__ == '__main__':
    js = json.load(open(source_file))
    create_embedding(js)
