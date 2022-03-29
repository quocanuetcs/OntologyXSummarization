from bleu import list_bleu

class BleuScore:
    def get_score(self, hypothesis, reference):
        return list_bleu([[reference]], [hypothesis]) / 100.0
