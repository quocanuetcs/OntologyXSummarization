import os
from stanfordcorenlp import StanfordCoreNLP

LIBRARY_PATH = os.path.dirname(os.path.realpath(__file__)) + '/stanford-corenlp-4.4.0'


class CoreNLP:

    def __init__(self, library_path=LIBRARY_PATH):
        if not os.path.isdir(library_path):
            os.system(os.path.dirname(os.path.dirname(os.path.realpath(__file__)) + "/downloader.sh"))
        self.core_nlp = StanfordCoreNLP(path_or_host=library_path, memory='4g')

    def pos_tag(self, tokenized):
        return self.core_nlp.pos_tag(tokenized)
