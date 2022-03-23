class BaseModel:
    def __init__(self):
        self.questions = None

    def train(self, questions):
        self.questions = questions

    def predict_sentence(self, question_id, answer_id, sentence_id):
        pass

    def predict_answer(self, question_id, answer_id):
        scores = list()
        for sentence_id, sentence in self.questions[question_id].answers[answer_id].sentences.items():
            scores.append({
                'question_id': question_id,
                'answer_id': answer_id,
                'sentence_id': sentence_id,
                'score': self.predict_sentence(question_id, answer_id, sentence_id)
            })
        return scores

    def predict_question(self, question_id):
        scores = list()
        for answer_id, answer in self.questions[question_id].answers.items():
            scores.extend(self.predict_answer(question_id, answer_id))
        return scores

    def predict(self):
        scores = list()
        for question_id, question in self.questions.items():
            scores.extend(self.predict_question(question_id))
        return scores
