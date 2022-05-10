import string
import re
from nltk.corpus import stopwords

def rr(text):
    text = re.sub(r'[,:.;]? - ([A-Z])', r'. \1', text)
    text = re.sub(r'[,:.;]? \t([A-Z])', r'. \1', text)
    return text

TAG_RE = re.compile(r'<[^>]+>')
rz = re.compile(r'([00])\w+')

def cleanhtml(text):
    if text is None:
        return None
    text = rr(text)
    text = re.sub(r'[^\x00-\x7F]', ' ', text)
    text = text.encode("ascii", errors="ignore").decode()
    text = rz.sub(' ', text)
    text = text.replace('\n', '')
    text = text.replace('\t', '')
    for i in range(0,10):
        text = text.replace('[{}]'.format(str(i)),'')
    return TAG_RE.sub(' ', text)

def remove_whitespaces(text):
    return " ".join(text.split()).lstrip().rstrip()

def remove_punctuations(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def lower_case(text):
    return text.lower()

def normalize(text):
    steps = [cleanhtml, lower_case, remove_punctuations, remove_whitespaces]
    for step in steps:
        text = step(text)
    return text

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [i for i in tokens if i not in stop_words]

if __name__ == '__main__':
    stop_words = set(stopwords.words('english'))
    print(stop_words)





