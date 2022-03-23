from nltk import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re
import nltk


def lower_case(text):
    return text.lower()


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
    return TAG_RE.sub(' ', text)


def remove_punctuations(text):
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_whitespaces(text):
    return " ".join(text.split())


def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [i for i in tokens if i not in stop_words]


def stemming(tokens):
    stemmer = PorterStemmer()
    result = []
    for word in tokens:
        result.append(stemmer.stem(word))
    return result


def lemmatization(tokens):
    lemmatizer = WordNetLemmatizer()
    result = []
    for word in tokens:
        result.append(lemmatizer.lemmatize(word))
    return result


def normalize(text):
    steps = [cleanhtml, lower_case, remove_punctuations, remove_whitespaces]
    for step in steps:
        text = step(text)
    return text


def tokenize(text):
    tokens = word_tokenize(text)
    steps = [remove_stopwords, stemming, lemmatization]
    for step in steps:
        tokens = step(tokens)
    return tokens


def remove_numbers(tokens):
    return [i for i in tokens if not i.isnumeric()]


def is_connect(text):
    text = text.lower()
    l = ['to', 'the', 'a', 'an', 'in', 'by', 'and', 'or', 'that', 'as', 'like', 'of', "'s"]

    return text in l


def is_upcase(text):
    upcase = 'QWERTYUIOPASDGJKLZXCVBNM'
    whole_text = True
    if len(text) == 1:
        whole_text = False
    for i in text:
        if i not in upcase:
            whole_text = False
    return text[0] in upcase and not whole_text


def resplit(text):
    tokens = word_tokenize(text)
    tokens = nltk.pos_tag(tokens)

    sen = []
    for tok, type in tokens:
        if len(sen) == 0:
            sen.append([tok])
        else:
            split_point = is_upcase(tok)
            if (is_upcase(sen[-1][-1]) or is_connect(sen[-1][-1])) and type not in ['RB', 'NN']:
                split_point = False
            if split_point:
                sen.append([tok])
            else:
                sen[-1].append(tok)
    return [' '.join(i) for i in sen]

#
# a = "TableSome Causes and Features of Memory Loss Cause Common Features* Tests Age-related memory changes (age-associated memory impairment) Occasional forgetfulness of such things as names or the location of car keys No effect on thinking, other mental functions, or the ability to do daily activities A doctor's examination (particularly a neurologic examination and mental status testing to assess functions such as attention, orientation, and memory) Mild cognitive impairment Memory loss that is more severe than expected for a person's age, particularly difficulty remembering recent events and conversations (short-term memory loss) No effect on the ability to do daily activities An increased risk of developing dementia A doctor's examination Sometimes formal neuropsychologic testing, which resembles mental status testing but evaluates function in more detail Dementia Memory loss that becomes worse as time passes, eventually with no awareness of the loss Difficulty using and understanding language, doing usual manual tasks, thinking, and planning (for example, planning and shopping for meals), resulting in not being able to function normally Disorientation (for example, not knowing the time or location) Difficulty recognizing faces or common objects Changes in personality or behavior (for example, becoming irritable, agitated, paranoid, inflexible, or disruptive) A doctors examination Usually MRI or CT of the brain Sometimes formal neuropsychologic testing Possibly a spinal tap (lumbar puncture) to measure levels of two abnormal proteins (amyloid and tau) that occur in Alzheimer disease Sometimes blood tests to check for certain causes, such as an underactive thyroid gland (hypothyroidism) or a vitamin deficiency Depression Memory loss and awareness of the loss, usually accompanied by intense sadness, and lack of interest in usual pleasures Sometimes sleep problems (too little or too much), loss of appetite, and slowing of thinking, speech, and general activity Common among people with dementia, mild cognitive impairment, or age-related changes in memory A doctors examination Sometimes use of standardized questionnaires to identify depression Drugs, such as Drugs with anticholinergic effects, including some antidepressants and many antihistamines (used in OTC sleep aids, cold remedies, and allergy drugs) Opioids Drugs that help people sleep (sedatives) Use of a drug that can cause memory loss Often recent use of a new drug, an increase in a drugs dose, or a change in health that prevents the drug from being processed and eliminated from the body normally, as can occur in kidney or liver disorders Typically stopping the drug to see whether memory improves *Features include symptoms and results of the doctor's examination."
# for sen in resplit(a):
#     print('__ '+sen)
