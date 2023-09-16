import re 
import gensim

import underthesea
import pandas as pd 

from time import perf_counter

class ModelGensim:

  def __init__(self, path_stopword):
    # preprocess
    self.stop_word = self.getstopword(path_stopword)
    self.re_number = re.compile(r"[a-z]*[0-9]+[.,]?[0-9]*[a-z]*")
    self.re_special = re.compile(r'[!\'"#$%&()*+<=>?,-./:;@[\\]^_`{|}~\t\n]')
    self.re_word = re.compile(r'[^aàáảãạâầấẩẫậâăằắẳẵặăeèéẻẽẹêềếểễệêiìíỉĩịoòóỏõọồốôổỗộôơờớởỡợơuùúủũụưừứửữựưyỳýỷỹỵđa-z0-9\s]')
    self.re_space = re.compile(r'\s\s+')

    # model
    self.corpus = None
    self.word_dictnary = None
    self.tf_idf = None
    self.sparse_matrix_similarity = None


  def getstopword(self, path):
    with open(path, 'r', encoding ='utf-8') as file:
      stop_words = file.read()
      stop_words = stop_words.split("\n")

    return set(stop_words)

  def preprocess_text(self, text):
    # lower
    text = text.lower()
    # mask number
    text = self.re_number.sub(" ", text)
    # charater special
    text = self.re_special.sub(" ",text)
    # drop charater word wrong form
    text = self.re_word.sub("",text)
    # drop duplicate space
    text = self.re_space.sub(" ",text)
    return text

  def tokenizer(self, data):
    result = []
    for x in data:
      save = underthesea.word_tokenize(x, format='text').split(" ")
      save = [x for x in save if x not in self.stop_word]
      result.append(save)
    return result

  def train(self,data):
    # preprocess data
    sta = perf_counter()
    new_data = [self.preprocess_text(x) for x in data]
    print("Pre-process: Done; Time:", round(perf_counter()-sta,2), "s")

    # tokeinzer
    sta = perf_counter()
    data_tokenizer = self.tokenizer(new_data)
    print("Tokenizer: Done; Time:", round(perf_counter()-sta,2), "s")

    # Word embedding
    sta = perf_counter()
    self.word_dictionary =  gensim.corpora.Dictionary(data_tokenizer)
    self.corpus = [self.word_dictionary.doc2bow(text) for text in data_tokenizer]
    self.tf_idf = gensim.models.TfidfModel(self.corpus)
    print("Embeding: Done; Time:", round(perf_counter()-sta,2), "s")

    # Training
    sta = perf_counter()
    self.matrix_similarity = gensim.similarities.SparseMatrixSimilarity(self.tf_idf[self.corpus],
                                                                        num_features = len(self.word_dictionary))
    print("Training: Done; Time:", round(perf_counter()-sta,2), "s")
    return None

  def predict(self,input,df):
    input = self.preprocess_text(input)
    input = self.tokenizer([input])
    vec = self.word_dictionary.doc2bow(input[0])
    sim = self.matrix_similarity[self.tf_idf[vec]]
    df_new  = df[df.columns.tolist()]
    df_new["score"] = sim.tolist()
    df_new = df_new.sort_values(by="score",ascending=False)
    return df_new
  
if __name__ == "__main__":
  df_train = pd.read_csv("data/ProductRaw.csv")
  df_train = df_train[['item_id', 'name', 'description', 'group']]
  df_train = df_train.dropna()
  df_train["name_view_group"] =  df_train['name'] + ' ' + df_train['description'] + ' ' + df_train['group']

  path_stopword = "data/vietnamese-stopwords.txt"
  model = ModelGensim(path_stopword)

  model.train(df_train["name_view_group"].tolist())

  import pickle
  with open('checkpoint/model_gensim.pkl', 'wb') as fs:
    pickle.dump(model, fs, pickle.HIGHEST_PROTOCOL)
