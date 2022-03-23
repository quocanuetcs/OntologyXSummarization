from models.base_model import BaseModel
import random


class RandomScore(BaseModel):
    def predict_sentence(self, question_id, answer_id, sentence_id):
        return random.random()

