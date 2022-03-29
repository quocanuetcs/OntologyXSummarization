import json
import xlrd
from utils.logger import get_logger
from utils.preprocessing import cleanhtml
import os

VAL_QUES_FILE_NAME = "Validation/MEDIQA2021_Task2_ValidationSet_ShortQuestions.txt"
VAL_ANSWER_FILE_NAME = 'Validation/MEDIQA2021_Task2_ValidationSet_Answers.xls'
VAL_EXT_MULTISUMM_FILE_NAME = 'Validation/MEDIQA2021_Task2_ValidationSet_MultiExtractiveSummaries.txt'
VAL_ABS_MULTISUMM_FILE_NAME = 'Validation/MEDIQA2021_Task2_ValidationSet_MultiAbstrativeSummaries.txt'

TEST_QUES_FILE_NAME = "Test/MEDIQA2021_Task2_MAS_TestSet_shortQuestions.txt"
TEST_ANSWER_FILE_NAME = 'Test/MEDIQA2021_Task2_MAS_TestSet_Answers.xls'
TEST_EXT_MULTISUMM_FILE_NAME = 'Test/MEDIQA2021_Task2_TestSet_MultiExtractiveSummaries.txt'
TEST_ABS_MULTISUMM_FILE_NAME = 'Test/MEDIQA2021_Task2_TestSet_MultiAbstrativeSummaries.txt'

def process_answers(file_path):
    wb = xlrd.open_workbook(file_path)
    sheet = wb.sheets()[0]
    # sheet = wb.sheet_by_name('MEDIQA2021_Task2_ValidationSet_')
    rs = dict()
    keys = ['ques_id', 'ans_id', 'article']
    for i in range(1, sheet.nrows):
        a = dict()
        m = sheet.row_values(i)
        q_id = str(int(m[0]))
        a_id = m[1]
        ar = m[2]
        if q_id in rs:
            rs[q_id][a_id] = dict()
            rs[q_id][a_id]['answer_abs_summ'] = None
            rs[q_id][a_id]['answer_ext_summ'] = None
            rs[q_id][a_id]['section'] = None
            rs[q_id][a_id]['url'] = None
            rs[q_id][a_id]['rating'] = None
            rs[q_id][a_id]['article'] = cleanhtml(ar)

        else:
            rs[q_id] = dict()
            rs[q_id][a_id] = dict()
            rs[q_id][a_id]['answer_abs_summ'] = None
            rs[q_id][a_id]['answer_ext_summ'] = None
            rs[q_id][a_id]['section'] = None
            rs[q_id][a_id]['url'] = None
            rs[q_id][a_id]['rating'] = None
            rs[q_id][a_id]['article'] = cleanhtml(ar)

    return rs


def split__(file_path):
    try:
        f = open(file_path, 'r', encoding='utf8')
        rs = dict()
        for line in f:
            if line[-1] == '\n':
                line = line[0:-1]
            s = line.split('||')
            rs[s[0]] = ' '.join(s[1:-1]) if len(s) > 2 else s[1]
    except:
        rs = None
    return rs


def join__(ans, ext, abs, ques):
    rs = dict()
    for ques_id, ques in ques.items():
        q = dict()
        q['question'] = ques
        q['answers'] = ans[ques_id]
        if abs is None:
            q['multi_abs_summ'] = None
        else:
            q['multi_abs_summ'] = abs[ques_id]
        if ext is None:
            q['multi_ext_summ'] = None
        else:
            q['multi_ext_summ'] = ext[ques_id]
        rs[ques_id] = q
    return rs


def from_in_input_data(name, input_path='../data/raw/'):

    source_file_name = ""
    answer_file_name = ""
    ext_multisumm_file_name = ""
    ques_file_name = ""
    abs_multisumm_file_name = ""
    out_path = '../data/raw_json/'

    if name=='validation':
        answer_file_name = VAL_ANSWER_FILE_NAME
        ext_multisumm_file_name = VAL_EXT_MULTISUMM_FILE_NAME
        ques_file_name = VAL_QUES_FILE_NAME
        abs_multisumm_file_name = VAL_ABS_MULTISUMM_FILE_NAME
        source_file_name = 'validation_raw.json'
    elif name=='test':
        answer_file_name = TEST_ANSWER_FILE_NAME
        ext_multisumm_file_name = TEST_EXT_MULTISUMM_FILE_NAME
        ques_file_name = TEST_QUES_FILE_NAME
        abs_multisumm_file_name = TEST_ABS_MULTISUMM_FILE_NAME
        source_file_name = 'test_raw.json'

    try:
        rs = json.load(open(out_path + source_file_name, 'r', encoding='utf8'))
        get_logger(__file__).info('{} found! Input data loaded from this directory!'.format(out_path + source_file_name))
    except FileNotFoundError:
        ans = process_answers(input_path + answer_file_name)
        ext = split__(input_path + ext_multisumm_file_name)
        ques = split__(input_path + ques_file_name)
        abs_ = split__(input_path + abs_multisumm_file_name)
        js = join__(ans, ext, abs_, ques)
        rs = js
        get_logger(__file__).info(
            '{}\n{}\n{}\n{} found!\nInput data created from this directory!'.format(input_path + answer_file_name,
                                                                                    input_path + ext_multisumm_file_name,
                                                                                    input_path + abs_multisumm_file_name,
                                                                                    input_path + ques_file_name))
        json.dump(js, open(out_path + source_file_name, 'w+', encoding='utf8'), indent=4, sort_keys=True)
        get_logger(__file__).info('Input data has been stored at {}'.format(out_path + source_file_name))
    return rs



