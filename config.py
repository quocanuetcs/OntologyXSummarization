class TFIDF_WEIGHT:
    def __init__(self):
        self.ratio_threshold = 0.7
        self.keywords_boost = 2.0
        self.keywords_type = 'weight'

class LEXRANK_WEIGHT:
    def __init__(self):
        self.threshold = 0.065
        self.tf_for_all_question = True

class NER_WEIGHT:
    def __init__(self):
        self.ner_threshold = 0.8
        self.word_threshold = 0.8

class FINAL_WEIGHT:
    def __init__(self):
        self.tfidf = 4
        self.query_based = 3
        self.lexrank = 2
        self.ner = 2
        self.wRWMD = 3
        self.total_weight = self.tfidf + self.lexrank + self.ner + self.query_based + self.wRWMD

class SENTENCE_SCORING:
    def __init__(self):
        self.tfidf = TFIDF_WEIGHT()
        self.lexrank = LEXRANK_WEIGHT()
        self.ner = NER_WEIGHT()
        self.final = FINAL_WEIGHT()


class SINGLE_SUM_FOR_SINGLE():
    def __init__(self):
        self.limit = 5
        self.limit_type = 'num'
        self.score_type = 'final'
        self.threshold = None

class SINGLE_SUM_FOR_MULTI():
    def __init__(self, limit=10, limit_type='num', score_type='final', threshold=None):
        self.limit = limit
        self.limit_type = limit_type
        self.score_type = score_type
        self.threshold = threshold

class MESH_COFIG():
    def __init__(self):
        self.A = False #Giai phau hoc
        self.B = False #Sinh vật nói chung
        self.C = True  #Bệnh
        self.D = True  #Thuốc, chất hóa học
        self.E = False  #Kỹ thuật và Thiết bị Phân tích, Chẩn đoán và Trị liệu
        self.F = False  #Tâm thần học và tâm lí học
        self.G = False  #Hiện tượng, quá trình
        self.H = False  #Luật và nghê nghiệp
        self.I = False #Nhân chủng học, Giáo dục, Xã hội học và Hiện tượng xã hội
        self.J = False  #Công nghệ, Công nghiệp và Nông nghiệp
        self.K = False #Nhân văn
        self.L = False  #Khoa học thông tin
        self.M = False #Ngườo
        self.N = False #Chăm sóc sức khỏe
        self.V = False #Đặc điểm xuất bản
        self.X = False #Check tag
        self.Z = False  #Vị trí địa lí
        self.R = False #Danh mục tác dụng dược lí
        self.Y = False #Danh mục tiêu đề phục

class ONTOLOGY_CONFIG():
    def __init__(self):
        self.have_mondo=True
        self.have_symp=False
        self.have_gene=False
        self.have_chemicals_diseases_relation=True


class WORD_CONFIG():
    def __init__(self):
        self.have_form = True
        self.have_derived_forms = True

class WORDNET():
    def __init__(self):
       self.word = WORD_CONFIG()


