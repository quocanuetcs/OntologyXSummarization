class BaseModel:
    def __init__(self):
        self.questions = None

    def train(self, questions):
        self.questions = questions

    def predict_sentence(self, question_id, answer_id, sentence_id):
        pass