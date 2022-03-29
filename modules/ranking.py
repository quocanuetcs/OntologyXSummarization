from copy import deepcopy
from json import JSONEncoder

from models.mmr.run_mmr import export_multiple_summaries
from utils.similarity import word_weight_base, bert_base
from pandas import DataFrame
from evaluations.bleu_evaluation import BleuScore
from evaluations.rouge_evaluation import RougeScore
from models.neighbors.neighbor_boost import NeighborBoost
from models.random.random_range_score import RandomRangeScore
from models.random.random_score import RandomScore
from models.tfidf.tfidf_score import TfIdfScore
from models.clustering.split_model import SplitParaScore
from models.lexrank.lexrank_score import LexrankScore
from models.querybase.querybase_score import QueryBaseCore, Answer
from models.graphbase.graph_base_model import GraphScore
from models.ner_ratio.ner_ratio import NerRatio
from utils import preprocessing
import os
import json
from utils.logger import get_logger

logger = get_logger(__file__)


class ObjectEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__


class Rank:

    def __init__(self,
                 question=None,
                 answer=None,
                 sentence=None):
        self.question_id = question.id
        self.question = question
        self.answer_id = answer.id
        self.answer = answer
        self.sentence_id = sentence.id
        self.sentence = sentence

        self.ques = int(''.join([i for i in str(question.id) if i in '1234567890']))
        self.ans = int(''.join([i for i in str(answer.id) if i in '1234567890']))
        self.sen = int(''.join([i for i in str(sentence.id) if i in '1234567890']))

        self.tfidf_score = None
        self.lexrank_score = None
        self.textrank_score = None
        self.query_base_score = None
        self.graph_score = None
        self.ner_score = None
        self.split_para_score = None
        self.final_score = None

        # Random scores
        self.random_range_score = None
        self.random_score = None

        # ROUGE evaluation
        self.precision_rouge_1 = None
        self.precision_rouge_2 = None
        self.precision_rouge_l = None

        # BLEU evaluation
        self.bleu = None

    def add_from_json(self, js):
        if 'tfidf_score' in js:
            self.tfidf_score = js['tfidf_score']
        if 'lexrank_score' in js:
            self.lexrank_score = js['lexrank_score']
        if 'textrank_score' in js:
            self.textrank_score = js['textrank_score']
        if 'query_base_score' in js:
            self.query_base_score = js['query_base_score']
        if 'graph_score' in js:
            self.graph_score = js['graph_score']
        if 'ner_score' in js:
            self.ner_score = js['ner_score']
        if 'split_para_score' in js:
            self.split_para_score = js['split_para_score']
        if 'final_score' in js:
            self.final_score = js['final_score']

    def add_tfidf_score(self, tfidf_score):
        self.tfidf_score = tfidf_score.predict_sentence(self.question_id, self.answer_id, self.sentence_id)
        return self

    def add_ner_score(self, model):
        self.ner_score = model.predict_sentence(self.question_id, self.answer_id, self.sentence_id)
        return self

    def add_lexrank_score(self, lexrank_score):
        self.lexrank_score = lexrank_score.predict_sentence(self.question_id, self.answer_id, self.sentence_id)
        return self

    def add_textrank_score(self, textrank_score):
        self.textrank_score = textrank_score.predict_sentence(self.question_id, self.answer_id, self.sentence_id)
        return self

    def add_query_base_score(self, model):
        self.query_base_score = model.predict_sentence(self.question_id, self.answer_id, self.sentence_id)
        return self

    def add_graph_score(self, model):
        self.graph_score = model.predict_sentence(self.question_id, self.answer_id, self.sentence_id)
        return self

    def add_split_para_score(self, model):
        self.split_para_score = model.predict_sentence(self.question_id, self.answer_id, self.sentence_id)
        return self

    def add_neighbor_boost(self, model, score_type=None):
        score = model.predict_sentence(self.question_id, self.answer_id, self.sentence_id)
        if score_type == 'final':
            self.final_score = score
        elif score_type == 'tfidf':
            self.tfidf_score = score
        elif score_type == 'lexrank':
            self.lexrank_score = score
        elif score_type == 'textrank':
            self.textrank_score = score
        elif score_type == 'query_base':
            self.query_base_score = score
        elif score_type == 'graph_base':
            self.graph_score = score
        elif score_type == 'ner':
            self.ner_score = score
        elif score_type == 'split_para':
            self.split_para_score = score
        elif score_type == 'random':
            self.random_score = score
        elif score_type == 'random_range':
            self.random_range_score = score
        return self

    def add_final_by_lstm(self, model):
        self.final_score = model.predict_sentence(self.question_id, self.answer_id, self.sentence_id)
        return self

    def add_final_score(self,
                        final_score_type=None,
                        tfidf=None,
                        lexrank=None,
                        textrank=None,
                        graph=None,
                        query_base=None,
                        ner=None,
                        split_para=None):
        self.final_score = 0
        if final_score_type == 'sum':
            max_score = 0
            if self.tfidf_score is not None:
                self.final_score += self.tfidf_score * tfidf
                max_score += tfidf
            if self.lexrank_score is not None:
                self.final_score += self.lexrank_score * lexrank
                max_score += lexrank
            if self.textrank_score is not None:
                self.final_score += self.textrank_score * textrank
                max_score += textrank
            if self.query_base_score is not None:
                self.final_score += self.query_base_score * query_base
                max_score += query_base
            if self.graph_score is not None:
                self.final_score += self.graph_score * graph
                max_score += graph
            if self.ner_score is not None:
                self.final_score += self.ner_score * ner
                max_score += ner
            if self.split_para_score is not None:
                self.final_score += self.split_para_score * split_para
                max_score += split_para
            # Normalize score
            self.final_score /= max_score
        else:
            if tfidf == 1:
                self.final_score = max(self.final_score, self.tfidf_score)
            if lexrank == 1:
                self.final_score = max(self.final_score, self.lexrank_score)
            if textrank == 1:
                self.final_score = max(self.final_score, self.textrank_score)
            if query_base == 1:
                self.final_score = max(self.final_score, self.query_base_score)
            if graph == 1:
                self.final_score = max(self.final_score, self.graph_score)
            if ner == 1:
                self.final_score = max(self.final_score, self.ner_score)
            if split_para == 1:
                self.final_score = max(self.final_score, self.split_para_score)
        return self

    def add_random_range_score(self, random_range_score):
        self.random_range_score = random_range_score.predict_sentence(self.question_id, self.answer_id,
                                                                      self.sentence_id)
        return self

    def add_random_score(self, random_score):
        self.random_score = random_score.predict_sentence(self.question_id, self.answer_id, self.sentence_id)
        return self

    def add_rouge_evaluation(self, summary_type):
        if summary_type == 'single':
            score = RougeScore().get_score(self.sentence.sentence, self.answer.answer_ext_summ)
        else:
            score = RougeScore().get_score(self.sentence.sentence, self.question.multi_ext_summ)
        self.precision_rouge_1 = score['rouge-1']['p']
        self.precision_rouge_1 = score['rouge-1']['p']
        self.precision_rouge_2 = score['rouge-2']['p']
        self.precision_rouge_l = score['rouge-l']['p']
        return self

    def add_bleu_evaluation(self):
        self.bleu = BleuScore().get_score(self.sentence.sentence, self.answer.answer_ext_summ)
        return self

    def to_x_y(self):
        x = [self.query_base_score, self.split_para_score, self.tfidf_score, self.ner_score, self.textrank_score,
             self.lexrank_score]
        y = self.precision_rouge_2


class Ranking:

    def reload_df(self):
        for question_id, ranks in self.ranks.items():
            self.ranks_df[question_id] = DataFrame([o.__dict__ for o in ranks])

    def __init__(self, questions, name="preprocessing_v3"):
        self.name = name
        self.questions = questions
        self.ranks = dict()
        self.ranks_df = dict()
        for question_id, question in questions.items():
            self.ranks[question_id] = list()
            for answer_id, answer in question.answers.items():
                for sentence_id, sentence in answer.sentences.items():
                    self.ranks[question_id].append(Rank(question=question, answer=answer, sentence=sentence))
        self.reload_df()

    def add_tfidf_score(self, tfidf_ratio_threshold=None, keywords_boost=None, keywords_type=None):
        tfidf_score = TfIdfScore(tfidf_ratio_threshold=tfidf_ratio_threshold,
                                 keywords_boost=keywords_boost,
                                 keywords_type=keywords_type)
        tfidf_score.train(self.questions)
        for question_id, ranks in self.ranks.items():
            logger.info('Add Tf-Idf score of question {}'.format(question_id))
            for rank in ranks:
                rank.add_tfidf_score(tfidf_score)
        self.reload_df()
        return self

    def add_lexrank_score(self, threshold=None, tf_for_all_question=None):
        lexrank_score = LexrankScore(threshold=threshold, tf_for_all_question=tf_for_all_question)
        lexrank_score.train(self.questions)
        for question_id, ranks in self.ranks.items():
            logger.info('Add Lexrank score of question {}'.format(question_id))
            for rank in ranks:
                rank.add_lexrank_score(lexrank_score)
        self.reload_df()
        return self

    # def add_textrank_score(self, phrases_ratio=None):
    #     textrank_score = TextRankScore(phrases_ratio=phrases_ratio)
    #     textrank_score.train(self.questions)
    #     for question_id, ranks in self.ranks.items():
    #         logger.info('Add Textrank score of question {}'.format(question_id))
    #         for rank in ranks:
    #             rank.add_textrank_score(textrank_score)
    #     self.reload_df()
    #     return self

    def add_query_base_score(self, sim=bert_base, ner_weight=None):
        score = QueryBaseCore(ner_weight=ner_weight, sim=sim)
        score.train(self.questions)
        for question_id, ranks in self.ranks.items():
            logger.info('Add Query-based score of question {}'.format(question_id))
            for rank in ranks:
                rank.add_query_base_score(score)
        self.reload_df()
        return self

    def add_graph_score(self, sim=word_weight_base, ner_weight=None):
        score = GraphScore(ner_weight=ner_weight, sim=sim)
        score.train(self.questions)
        for question_id, ranks in self.ranks.items():
            logger.info('Add Graph-based score of question {}'.format(question_id))
            for rank in ranks:
                rank.add_graph_score(score)
        self.reload_df()
        return self

    def add_ner_score(self, ner_threshold=None, word_threshold=None):
        score = NerRatio(ner_threshold=ner_threshold, word_threshold=word_threshold)
        score.train(self.questions)
        for question_id, ranks in self.ranks.items():
            logger.info('Add NER score of question {}'.format(question_id))
            for rank in ranks:
                rank.add_ner_score(score)
        self.reload_df()
        return self

    def add_split_para_score(self, sim=bert_base, threshold=None):
        score = SplitParaScore(threshold=threshold, sim=sim)
        score.train(self.questions)
        for question_id, ranks in self.ranks.items():
            logger.info('Add Split-parascore of question {}'.format(question_id))
            for rank in ranks:
                rank.add_split_para_score(score)
        self.reload_df()
        return self

    def add_neighbor_boost(self,
                           score_type=None,
                           neighbor_type=None,
                           limit_range=None,
                           relative_range=None,
                           threshold=None,
                           boost_first=None,
                           boost_last=None):
        neighbor_score = NeighborBoost(limit_range=limit_range,
                                       relative_range=relative_range,
                                       score_type=score_type,
                                       neighbor_type=neighbor_type,
                                       threshold=threshold,
                                       boost_first=boost_first,
                                       boost_last=boost_last)
        neighbor_score.train(self.ranks)
        for question_id, ranks in self.ranks.items():
            logger.info('Add Neighbor boost of question {}'.format(question_id))
            for rank in ranks:
                rank.add_neighbor_boost(neighbor_score, score_type)
        self.reload_df()
        return self

    def add_final_score(self,
                        final_score_type='sum',
                        tfidf=7,
                        lexrank=3,
                        textrank=1,
                        graph=3,
                        query_base=5,
                        ner=5,
                        split_para=10):
        for question_id, ranks in self.ranks.items():
            logger.info('Add Final score of question {}'.format(question_id))
            max_score, min_score = 0, 1000000000
            for rank in ranks:
                rank.add_final_score(final_score_type=final_score_type,
                                     tfidf=tfidf,
                                     lexrank=lexrank,
                                     textrank=textrank,
                                     graph=graph,
                                     query_base=query_base,
                                     ner=ner,
                                     split_para=split_para)
                max_score = max(max_score, rank.final_score)
                min_score = min(min_score, rank.final_score)
            # Normalize score
            for rank in ranks:
                if max_score == min_score:
                    rank.final_score = 1
                else:
                    rank.final_score = (rank.final_score - min_score) / (max_score - min_score)
        self.reload_df()
        return self

    def add_random_score(self, num=None):
        random_score = RandomScore(num=num)
        for question_id, ranks in self.ranks.items():
            logger.info('Add Random score of question {}'.format(question_id))
            for rank in ranks:
                rank.add_random_score(random_score)
        self.reload_df()
        return self

    def add_random_range_score(self, num=None, length=None):
        random_range_score = RandomRangeScore(num=num, length=length)
        random_range_score.train(self.questions)
        for question_id, ranks in self.ranks.items():
            for rank in ranks:
                rank.add_random_range_score(random_range_score)
        self.reload_df()
        return self

    def add_all_score(self, sims=None):
        if sims is None:
            sims = [bert_base, word_weight_base, bert_base]
        if os.path.exists(
                os.path.dirname(os.path.realpath(__file__)) + '/../data/ranking/' + self.name + '_scores.json'):
            self.load_scores()
            return self
        #self.add_textrank_score()
        self.add_ner_score()
        self.add_tfidf_score()
        self.add_lexrank_score()
        self.add_query_base_score(sims[0])
        self.add_graph_score(sims[1])
        self.add_split_para_score(sims[2])
        self.add_final_score()
        self.export_scores()
        return self

    def add_rouge_evaluation(self, summary_type='single'):
        for question_id, ranks in self.ranks.items():
            logger.info('Add ROUGE evaluation of question {}'.format(question_id))
            for rank in ranks:
                rank.add_rouge_evaluation(summary_type)
        self.reload_df()
        return self

    def get_score(self, obj, score_type='final'):
        if score_type == 'final':
            return obj.final_score
        elif score_type == 'tfidf':
            return obj.tfidf_score
        elif score_type == 'lexrank':
            return obj.lexrank_score
        elif score_type == 'textrank':
            return obj.textrank_score
        elif score_type == 'query_base':
            return obj.query_base_score
        elif score_type == 'graph_base':
            return obj.graph_score
        elif score_type == 'ner':
            return obj.ner_score
        elif score_type == 'split_para':
            return obj.split_para_score
        elif score_type == 'random':
            return obj.random_score
        elif score_type == 'random_range':
            return obj.random_range_score
        return None

    def export_scores(self, file_path=None, name=None):
        if name is None:
            name = self.name
        if file_path is None:
            file_path = os.path.dirname(os.path.realpath(__file__)) + '/../data/ranking/' + name + '_scores.json'
        result = dict()
        for question_id, ranks in self.ranks.items():
            result[question_id] = dict()
            for rank in ranks:
                if rank.answer_id not in result[question_id]:
                    result[question_id][rank.answer_id] = dict()
                result[question_id][rank.answer_id][rank.sentence_id] = {
                    'tfidf_score': rank.tfidf_score,
                    'lexrank_score': rank.lexrank_score,
                    'textrank_score': rank.textrank_score,
                    'query_base_score': rank.query_base_score,
                    'graph_score': rank.graph_score,
                    'ner_score': rank.ner_score,
                    'split_para_score': rank.split_para_score,
                    'final_score': rank.final_score,
                    'precision_rouge_1': rank.precision_rouge_1,
                    'precision_rouge_2': rank.precision_rouge_2,
                    'precision_rouge_l': rank.precision_rouge_l
                }
        with open(file_path, 'w') as file:
            file.write(json.dumps(result, cls=ObjectEncoder, indent=2, sort_keys=True))
        return result

    def load_scores(self, file_path=None, name=None):
        if name is None:
            name = self.name
        if file_path is None:
            file_path = os.path.dirname(os.path.realpath(__file__)) + '/../data/ranking/' + name + '_scores.json'
        with open(file_path) as file:
            data = json.load(file)
        for question_id, ranks in self.ranks.items():
            for rank in ranks:
                if data[question_id][rank.answer_id][rank.sentence_id]['tfidf_score'] is not None:
                    rank.tfidf_score = data[question_id][rank.answer_id][rank.sentence_id]['tfidf_score']
                if data[question_id][rank.answer_id][rank.sentence_id]['lexrank_score'] is not None:
                    rank.lexrank_score = data[question_id][rank.answer_id][rank.sentence_id]['lexrank_score']
                if data[question_id][rank.answer_id][rank.sentence_id]['textrank_score'] is not None:
                    rank.textrank_score = data[question_id][rank.answer_id][rank.sentence_id]['textrank_score']
                if data[question_id][rank.answer_id][rank.sentence_id]['query_base_score'] is not None:
                    rank.query_base_score = data[question_id][rank.answer_id][rank.sentence_id]['query_base_score']
                if data[question_id][rank.answer_id][rank.sentence_id]['graph_score'] is not None:
                    rank.graph_score = data[question_id][rank.answer_id][rank.sentence_id]['graph_score']
                if data[question_id][rank.answer_id][rank.sentence_id]['ner_score'] is not None:
                    rank.ner_score = data[question_id][rank.answer_id][rank.sentence_id]['ner_score']
                if data[question_id][rank.answer_id][rank.sentence_id]['split_para_score'] is not None:
                    rank.split_para_score = data[question_id][rank.answer_id][rank.sentence_id]['split_para_score']
                if data[question_id][rank.answer_id][rank.sentence_id]['final_score'] is not None:
                    rank.final_score = data[question_id][rank.answer_id][rank.sentence_id]['final_score']
                if data[question_id][rank.answer_id][rank.sentence_id]['precision_rouge_1'] is not None:
                    rank.precision_rouge_1 = data[question_id][rank.answer_id][rank.sentence_id][
                        'precision_rouge_1']
                if data[question_id][rank.answer_id][rank.sentence_id]['precision_rouge_2'] is not None:
                    rank.precision_rouge_2 = data[question_id][rank.answer_id][rank.sentence_id][
                        'precision_rouge_2']
                if data[question_id][rank.answer_id][rank.sentence_id]['precision_rouge_l'] is not None:
                    rank.precision_rouge_l = data[question_id][rank.answer_id][rank.sentence_id][
                        'precision_rouge_l']
        self.reload_df()
        return self

    def export_single_summ(self, limit=5,
                           limit_type='num',
                           score_type='final',
                           to_text=False,
                           limit_o='sentence',
                           has_bleu=False):
        rs = {}
        for question_id, all_ranks in self.ranks.items():
            # Divide into answers
            rs_a = {}
            ranks_by_answer = dict()

            for rank in all_ranks:
                if rank.answer_id not in ranks_by_answer:
                    ranks_by_answer[rank.answer_id] = list()
                ranks_by_answer[rank.answer_id].append(rank)

            for answer_id, ranks in ranks_by_answer.items():
                if len(ranks) == 0:
                    continue
                # Filter sentences
                sorted_ranks = sorted(ranks, key=lambda obj: self.get_score(obj, score_type), reverse=True)
                threshold_position = min(limit, len(sorted_ranks))
                if limit_type == 'ratio':
                    threshold_position = round(int(limit * len(sorted_ranks)))
                elif limit_type == 'threshold':
                    threshold_position = 0
                    for i in range(len(sorted_ranks)):
                        if self.get_score(sorted_ranks[i], score_type) < limit:
                            break
                        threshold_position += 1
                threshold_position = max(threshold_position, 1)

                if limit_o == 'sentence':
                    sorted_ranks = sorted_ranks[:threshold_position]
                else:
                    get_len = 0
                    for rank in sorted_ranks:
                        get_len += len(rank.sentence.tokens)
                    if limit_type == 'ratio':
                        get_len = limit * get_len
                    else:
                        get_len = limit
                    m = []
                    n_toks = 0
                    for rank in sorted_ranks:
                        if n_toks + len(rank.sentence.tokens) <= get_len:
                            m.append(rank)
                            n_toks += len(rank.sentence.tokens)
                        else:
                            break
                    sorted_ranks = m
                sorted_ranks = sorted(sorted_ranks, key=lambda obj: int(obj.sentence_id))
                if to_text:
                    rs_a[answer_id] = ' '.join([rank.sentence.sentence for rank in sorted_ranks]).strip()
                else:
                    rs_a[answer_id] = sorted_ranks
            rs[question_id] = rs_a
        # if to_text:
        #     rs = json.dumps(rs, indent=2, sort_keys=True)
        return rs

    def remove_redundancy(self, ranks):
        ranks = sorted(ranks, key=lambda obj: len(obj.sentence.sentence), reverse=True)
        redundant_sentences = dict()
        for i in range(len(ranks)):
            for j in range(i):
                score = RougeScore().get_score(ranks[i].sentence.sentence, ranks[j].sentence.sentence)
                if score['rouge-2']['p'] >= 0.8:
                    redundant_sentences[i] = True
                    break
                # if bert_base(ranks[i].sentence, ranks[j].sentence) >= 0.5:
                #     redundant_sentences[i] = True
                #     break
        result_ranks = list()
        for i in range(len(ranks)):
            if i not in redundant_sentences:
                result_ranks.append(ranks[i])
        result_ranks = sorted(result_ranks, key=lambda obj: int(obj.sentence_id))
        return result_ranks

    def summary(self,
                limit=6,
                limit_type='num',
                score_type='final',
                has_bleu=False,
                summary_type='single'):
        details = list()
        for question_id, all_ranks in self.ranks.items():
            # Divide into answers
            ranks_by_answer = dict()
            for rank in all_ranks:
                if rank.answer_id not in ranks_by_answer:
                    ranks_by_answer[rank.answer_id] = list()
                ranks_by_answer[rank.answer_id].append(rank)
            for answer_id, ranks in ranks_by_answer.items():
                if len(ranks) == 0:
                    continue
                # Filter sentences
                sorted_ranks = sorted(ranks, key=lambda obj: self.get_score(obj, score_type), reverse=True)
                threshold_position = min(limit, len(sorted_ranks))
                if limit_type == 'ratio':
                    threshold_position = round(int(limit * len(sorted_ranks)))
                elif limit_type == 'threshold':
                    threshold_position = 0
                    for i in range(len(sorted_ranks)):
                        if self.get_score(sorted_ranks[i], score_type) < limit:
                            break
                        threshold_position += 1
                threshold_position = max(threshold_position, 1)
                sorted_ranks = sorted_ranks[:threshold_position]
                sorted_ranks = sorted(sorted_ranks, key=lambda obj: int(obj.sentence_id))
                sorted_ranks = self.remove_redundancy(sorted_ranks)
                # Calculate scores
                predict_summary = ' '.join(rank.sentence.sentence for rank in sorted_ranks)
                predict_summary = preprocessing.cleanhtml(predict_summary)
                predict_sequence = [rank.sentence_id for rank in sorted_ranks]
                if summary_type == 'single':
                    actual_summary = sorted_ranks[0].answer.answer_ext_summ
                else:
                    actual_summary = sorted_ranks[0].question.multi_ext_summ
                actual_sequence = list()
                for rank in ranks:
                    if rank.precision_rouge_2 >= 0.7:
                        actual_sequence.append(rank.sentence_id)
                scores = RougeScore().get_score(predict_summary, actual_summary)
                details.append({
                    'question_id': question_id,
                    'answer_id': answer_id,
                    'predict_summary': predict_summary,
                    'predict_sequence': predict_sequence,
                    'actual_summary': actual_summary,
                    'actual_sequence': actual_sequence,
                    'rouge-1 precision': scores['rouge-1']['p'],
                    'rouge-1 recall': scores['rouge-1']['r'],
                    'rouge-1 f1': scores['rouge-1']['f'],
                    'rouge-2 precision': scores['rouge-2']['p'],
                    'rouge-2 recall': scores['rouge-2']['r'],
                    'rouge-2 f1': scores['rouge-2']['f'],
                    'rouge-l precision': scores['rouge-l']['p'],
                    'rouge-l recall': scores['rouge-l']['r'],
                    'rouge-l f1': scores['rouge-l']['f'],
                })
                if has_bleu:
                    details[len(details) - 1]['bleu'] = BleuScore().get_score(predict_summary, actual_summary)
        details_df = DataFrame().from_records(details)
        summary_df = DataFrame().from_dict({
            'p': {
                'rouge-1': details_df['rouge-1 precision'].mean(),
                'rouge-2': details_df['rouge-2 precision'].mean(),
                'rouge-l': details_df['rouge-l precision'].mean()
            },
            'r': {
                'rouge-1': details_df['rouge-1 recall'].mean(),
                'rouge-2': details_df['rouge-2 recall'].mean(),
                'rouge-l': details_df['rouge-l recall'].mean()
            },
            'f1': {
                'rouge-1': details_df['rouge-1 f1'].mean(),
                'rouge-2': details_df['rouge-2 f1'].mean(),
                'rouge-l': details_df['rouge-l f1'].mean()
            },
        })
        result = {
            'summary': summary_df,
            'details': details_df
        }
        if has_bleu:
            result['bleu'] = details_df['bleu'].mean()
        return result

    def find_rank(self, sentence):
        for rank in self.ranks[sentence.question_id]:
            if rank.answer_id == sentence.answer_id and rank.sentence_id == str(sentence.id):
                return rank
        return None

    def merge_multi_to_single(self, limit_type='num', limit=10, score_type='final', summary_type='multi'):
        summaries = self.summary(limit_type=limit_type,
                                 limit=limit,
                                 score_type=score_type,
                                 summary_type=summary_type)['details']
        summaries_cnt = 0
        save_ranks = dict()
        for question_id, question in self.questions.items():
            save_ranks[question_id] = list()
            sentences = dict()
            sentences_str = list()
            sentences_cnt = 0
            for answer_id, answer in question.answers.items():
                for sentence_id in summaries['predict_sequence'][summaries_cnt]:
                    sentences_cnt += 1
                    save_ranks[question_id].append(deepcopy(self.find_rank(answer.sentences[sentence_id])))

                    sentences[str(sentences_cnt)] = answer.sentences[sentence_id]
                    sentences[str(sentences_cnt)].answer_id = question_id + '_Answer1'
                    sentences[str(sentences_cnt)].id = str(sentences_cnt)
                    sentences_str.append(answer.sentences[sentence_id].sentence)

                    save_ranks[question_id][sentences_cnt - 1].sentence_id = str(sentences_cnt)
                    save_ranks[question_id][sentences_cnt - 1].sentence = sentences[str(sentences_cnt)]
                summaries_cnt += 1
            question.answers = dict()
            question.answers[question_id + '_Answer1'] = Answer(id=question_id + '_Answer1',
                                                                question_id=question_id,
                                                                article=' '.join(sentences_str),
                                                                section=' '.join(sentences_str),
                                                                answer_abs_summ=question.multi_abs_summ,
                                                                answer_ext_summ=question.multi_ext_summ)
            for rank in save_ranks[question_id]:
                rank.answer_id = question_id + '_Answer1'
                rank.answer = question.answers[question_id + '_Answer1']
                rank.question = question
            question.answers[question_id + '_Answer1'].sentences = sentences
        self.ranks = save_ranks
        # for question_id, question in self.questions.items():
        #     self.ranks[question_id] = list()
        #     for answer_id, answer in question.answers.items():
        #         for sentence_id, sentence in answer.sentences.items():
        #             self.ranks[question_id].append(Rank(question=question, answer=answer, sentence=sentence))
        self.reload_df()
        return self


def evaluate_multiple_ranking(ranking):
    """
    example running mmr
    """
    a = export_multiple_summaries(ranking, opt='from_single_summ', using_mmr='using_mmr')
    a = export_multiple_summaries(ranking, opt='not_from_single_summ', using_mmr='using_mmr')
    a = export_multiple_summaries(ranking, opt='from_single_summ', using_mmr='not_using_mmr')
    a = export_multiple_summaries(ranking, opt='not_from_single_summ', using_mmr='not_using_mmr')


if __name__ == '__main__':
    from utils.data_loader import QuestionLoader

    loader = QuestionLoader(name='preprocessing_validation').read_json(
        '../data/preprocessing/preprocessing_validation.json')
    ranking = Ranking(loader.questions, 'validation')
    ranking.load_scores()
    ranking.add_final_score(final_score_type='sum', tfidf=1, lexrank=0, textrank=0, query_base=1, graph=0, ner=1,
                            split_para=0)
    ranking.add_neighbor_boost(relative_range=(1, 1), score_type='final', neighbor_type='center')
    ranking.merge_multi_to_single(limit=6)
    ranking.add_tfidf_score(keywords_boost=3.0, tfidf_ratio_threshold=0.9, keywords_type='ner')
    ranking.add_query_base_score(ner_weight=0)
    ranking.add_final_score(final_score_type='sum', tfidf=1, lexrank=0, textrank=0, query_base=1, graph=0, ner=0,
                            split_para=0)
    ranking.add_rouge_evaluation(summary_type='multi')
    print(ranking.summary(limit=200))
