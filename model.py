import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.manifold import TSNE
import math
import re
import spacy

DATAFRAME_PATH = 'test_df.pkl'
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
nlp = spacy.load('en_core_web_sm',disable=['parser', 'ner'])

# Manually pick an uncommon word to search for 
SINGLE_WORD_1 = 'scatter'
SINGLE_WORD_2 = 'multiplication'
SINGLE_WORD_3 = '3-d'
SINGLE_WORD_4 = 'Gaussian'

SENTENCE_1 = 'Compute within class and between class scatter matrix'# Generic fragment of a piece of code
SENTENCE_2 = 'Calculation of multiplication series'
SENTENCE_3 = '3d plot dataset' # Defined in the method name, no description
SENTENCE_4 = 'Calculate Gaussian Kernel' # RBF Kernel defined, not Gaussian - can this be inferred somehow??


CODE_1 = '''
def get_scatter_matrices(self, A, B):
    n_features = np.shape(A)[1]
    labels = np.unique(B)

    scatter = np.empty((n_features, n_features))
    for label in labels:
        _A = A[B == label]
        scatter += (len(_A)) * calculate_covariance_matrix(_A)

    return scatter
'''

CODE_2 = '''
def generate_multiplication_series(numLst, num=15, cols=50):
    A = np.zeros([numLst, num, cols])
    B = np.zeros([numLst, num, cols])
    for j in range(numLst):
        series = np.linspace(start, start*np.random.randint, 8), num=num, dtype=int)
        A[j] = to_categorical(mult_ser, n_col=cols)
        B[j] = np.roll(A[i], -1, axis=0)
    return A, B
'''

CODE_3 = '''
def 3dPlot(A, B):
    B = misc._transform(B, dim=3)
    fig = plt.figure()
    plt.scatter(B[:, 0], B[:, 1], B[:, 2], c=y)
    plt.show()
'''

CODE_4 = '''
def gauss_kernel(x_1, x_2, gamma):
    dist = np.linalg.norm(x_1 - x_2) ** 2
    return np.exp(-gamma * dist)
'''



CODE_DESC_1 = '''
def get_scatter_matrices(self, A, B):
    """ within class scatter matrix """
    n_features = np.shape(A)[1]
    labels = np.unique(B)

    scatter = np.empty((n_features, n_features))
    for label in labels:
        _A = A[B == label]
'''

CODE_DESC_2 = '''
def generate_multiplication_series(numLst, num=15, cols=50):
    """ multiplication series """
    A = np.zeros([numLst, num, cols])
    for j in range(numLst):
        series = np.linspace(start, start*np.random.randint, 8), num=num, dtype=int)
        A[j] = to_categorical(mult_ser, n_col=cols)
'''

CODE_DESC_3 = '''
def 3d_Plot(A, B):
    """ plot 3d data """
    B = misc._transform(B, dim=3)
    plt.scatter(B[:, 0], B[:, 1], B[:, 2], c=A)
    plt.show()
'''

CODE_DESC_4 = '''
def gauss_kernel(x_1, x_2, gamma):
    """ Gauss Kernel """
    dist = np.linalg.norm(x_1 - x_2) ** 2
'''

def cluster_embeddings(df, n_clusters, display_num):
    from sklearn.cluster import AgglomerativeClustering
    concat_df = df.lemmatized_method_description + ' ' + df.lemmatized_method_comments.apply(lambda x: ' '.join(x))
    embeddings = embed(concat_df)
    embeddings = np.array(embeddings).tolist()
    clusterer = AgglomerativeClustering(n_clusters=n_clusters, affinity="euclidean", linkage="ward")
    clusters = clusterer.fit_predict(embeddings)
    clusterDf = pd.DataFrame({"text": concat_df, "cluster": clusters})
    return df.loc[clusterDf['cluster'] == display_num]


class ExactMatches:
    def __init__(self, df):
        self.df = df
        self.concat_df = self._preprocess(df)

    def _preprocess(self, df):
        return df.method_code + df.method_description + df.method_comments.apply(lambda x: ' '.join(x))        
    
    def exactWordMatches(self, word):
        word = word.strip()
        matches = []
        for idx, description in enumerate(self.concat_df):
            if word in description:
                matches.append((description, self.df.filename.iloc[idx]))
        return matches
    
    def firstNMatches(self, matches, N):
        results_ = []
        for i, match in enumerate(matches):
            results_.append('[{}] Match in the file: "{}"'.format(i, match[1]))
            if i >=N-1:
                break
        results_.append('...')
        return results_


# To-do: lemmatize input
class BaselineModel:
    def __init__(self, df):
        self.df = df
        concat_df = self._preprocess(df)
        self.dataset = tf.constant(concat_df)

    def _preprocess(self, df):
        return df.lemmatized_method_description + ' ' + df.lemmatized_method_comments.apply(lambda x: ' '.join(x))

    def lemmatize(self, query):
        lem = ' '.join([token.lemma_ for token in list(nlp(query)) if (token.is_stop==False) or (token.lemma in ('between', 'within'))])
        print(lem)
        return lem 

    def embed_texts(self, texts):
        embeddings = embed(texts)
        return np.array(embeddings).tolist()

    def embeddings_plot(self):
        embeddings = self.embed_texts(self.dataset)
        mapped_embeddings = TSNE(metric='cosine').fit_transform(embeddings)
        return mapped_embeddings

    def get_sts_benchmark(self, query):
        query = self.lemmatize(query)
        duplicate_query = [query]*len(self.dataset)
        sts_encode1 = tf.nn.l2_normalize(embed(tf.constant(self.dataset)), axis=1)
        sts_encode2 = tf.nn.l2_normalize(embed(tf.constant(duplicate_query)), axis=1)
        cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
        clip_cosine_similarities = tf.clip_by_value(cosine_similarities, -1.0, 1.0)
        scores = 1.0 - tf.acos(clip_cosine_similarities) / math.pi
        return scores

    def match_sentence(self, query, num):
        scores = self.get_sts_benchmark(query).numpy()
        scoresidx = tf.argsort(scores,axis=-1,direction='ASCENDING',stable=False,name=None)
        maxidxs = scoresidx[-num:][::-1].numpy()
        results = []
        code_results = []
        for i, maxidx in enumerate(maxidxs):
            results.append('[{}]: {} score\
              \nMatched method: "{}" in file: "{}"'.format(i, scores[maxidx], self.df.iloc[maxidx].method_name, self.df.iloc[maxidx].filename))
            code_results.append(self.df.method_code[maxidx])
        return results, code_results


class SourceCodeModel:
    def __init__(self, df):
        self.df = df
        self.df['cleaned_method_code'] = self.df['method_code'].apply(lambda x: self._preprocess(x))

    def _preprocess(self, data):
        cleaned_data = re.sub(r'\W+', ' ', data)
        cleaned_data = cleaned_data.replace("_", " ")
        return cleaned_data.strip()

    def embed_texts(self, texts):
        embeddings = embed(texts)
        return np.array(embeddings).tolist()

    def embeddings_plot(self):
        embeddings = self.embed_texts(self.dataset)
        mapped_embeddings = TSNE(metric='cosine').fit_transform(embeddings)
        return mapped_embeddings

    def get_sts_benchmark(self, query):
        query = self._preprocess(query)
        dataset = tf.constant(self.df.cleaned_method_code)
        EMBED_DATASET = tf.nn.l2_normalize(embed(tf.constant(dataset)), axis=1)

        duplicate_query = [query]*len(dataset)
        sts_encode2 = tf.nn.l2_normalize(embed(tf.constant(duplicate_query)), axis=1)
        cosine_similarities = tf.reduce_sum(tf.multiply(EMBED_DATASET, sts_encode2), axis=1)
        clip_cosine_similarities = tf.clip_by_value(cosine_similarities, -1.0, 1.0)
        scores = 1.0 - tf.acos(clip_cosine_similarities) / math.pi
        return scores

    def match_sentence(self, query, num):
        scores = self.get_sts_benchmark(query).numpy()
        scoresidx = tf.argsort(scores,axis=-1,direction='ASCENDING',stable=False,name=None)
        maxidxs = scoresidx[-num:][::-1].numpy()
        results = []
        code_results = []
        for i, maxidx in enumerate(maxidxs):
            results.append('[{}]: {} score\
              \nMatched method: "{}" in file: "{}"'.format(i, scores[maxidx], self.df.iloc[maxidx].method_name, self.df.iloc[maxidx].filename))
            code_results.append(self.df.method_code[maxidx])
        return results, code_results


class PseudoJointEmbedding():
    def __init__(self, df):
        self.df = df
        self.SModel = SourceCodeModel(df)
        self.DModel = BaselineModel(df)

    def codeDescriptionParse(self, incompleteCode):
        code_without_desc = re.sub(r'\"""(.|\n)*?\"""', ' ', incompleteCode)
        description = re.search(r'\"""(.|\n)*?\"""', incompleteCode)[0]
        return code_without_desc.strip(), description.strip()
    
    def get_sts_benchmark(self, query):
        code, description = self.codeDescriptionParse(query)
        code_scores = self.SModel.get_sts_benchmark(code).numpy()
        desc_scores = self.DModel.get_sts_benchmark(description).numpy()
        return code_scores, desc_scores

    def match_sentence(self, query, num, code_weighting=0.5):
        code_scores, desc_scores = self.get_sts_benchmark(query)
        scores = code_weighting*code_scores + (1-code_weighting)*desc_scores
        scoresidx = tf.argsort(scores,axis=-1,direction='ASCENDING',stable=False,name=None)
        maxidxs = scoresidx[-num:][::-1].numpy()
        results = []
        code_results = []
        for i, maxidx in enumerate(maxidxs):
            results.append('[{}]: {} score\
              \nMatched method: "{}" in file: "{}"'.format(i, scores[maxidx], self.df.iloc[maxidx].method_name, self.df.iloc[maxidx].filename))
            code_results.append(self.df.method_code[maxidx])
        return results, code_results


def main():
    df = pd.read_pickle(DATAFRAME_PATH)
    # ToDo: figure out a better way of using each comment separately!
    #dataset = tf.constant(concat_df)
    print(cluster_embeddings(df, 20, 1))

    #model = ExactMatches(df)
    #matches = model.exactWordMatches('scatter')
    #print(model.firstNMatches(matches, 5))
    
    #model = BaselineModel(df)
    #LEMMATIZED_SENTENCE_1 = 'within class and between class scatter matrix'
    #print(model.match_sentence(LEMMATIZED_SENTENCE_1, 3))

    #model = SourceCodeModel(df)
    #print(model.match_sentence(CODE_1, 3))
    """
    model = PseudoJointEmbedding(df)
    results,_ = model.match_sentence(CODE_1, 3)
    print("Code 1")
    print(results)
    results,_ = model.match_sentence(CODE_2, 3)
    print("Code 2")
    print(results)
    results,_ = model.match_sentence(CODE_3, 3)
    print("Code 3")
    print(results)
    results,_ = model.match_sentence(CODE_4, 3)
    print("Code 4")
    print(results)"""



if __name__ == "__main__":
    main()
