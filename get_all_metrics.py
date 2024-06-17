import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
import matplotlib.pyplot as plt

import spacy
from pprint import pprint
import pandas as pd
import os
import numpy as np

from tqdm import tqdm

# NLTK Stop words
from nltk.corpus import stopwords
from scipy.sparse import dok_matrix

import time
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform, jensenshannon, cosine, cdist
from tensorflow.keras.datasets import mnist

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])


def find_txt_files(directory):
    txt_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                txt_files.append(os.path.join(root, file))
    return txt_files


def read_txt_file(file_path):
    try:
        with open(file_path, 'r', encoding="utf8") as file:
            file_contents = file.read()
        return file_contents
    except FileNotFoundError:
        return "File not found."


def sent_to_words(sentences):
    for sentence in tqdm(sentences):
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    for sent in tqdm(texts):
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc])
    return texts_out


def gather_data(base_path):
    # category 1: Computer Science
    directory_path = os.path.join(base_path, "Computer_Science")
    file_list = find_txt_files(directory_path)

    names_ComputerScience = []
    file_contents_ComputerScience = []
    labels_ComputerScience = []
    for file in file_list:
        names_ComputerScience.append(file.split("\\")[-1])
        file_contents_ComputerScience.append(read_txt_file(file))
        labels_ComputerScience.append("Computer Science")

    # category 2: History
    directory_path = os.path.join(base_path, "History")
    file_list = find_txt_files(directory_path)

    names_History = []
    file_contents_History = []
    labels_History = []
    for file in file_list:
        names_History.append(file.split("\\")[-1])
        file_contents_History.append(read_txt_file(file))
        labels_History.append("History")

    # category 3: Maths
    directory_path = os.path.join(base_path, "Maths")
    file_list = find_txt_files(directory_path)

    names_Maths = []
    file_contents_Maths = []
    labels_Maths = []
    for file in file_list:
        names_Maths.append(file.split("\\")[-1])
        file_contents_Maths.append(read_txt_file(file))
        labels_Maths.append("Maths")

    # category 4: accounts
    directory_path = os.path.join(base_path, "accounts")
    file_list = find_txt_files(directory_path)

    names_accounts = []
    file_contents_accounts = []
    labels_accounts = []
    for file in file_list:
        names_accounts.append(file.split("\\")[-1])
        file_contents_accounts.append(read_txt_file(file))
        labels_accounts.append("accounts")

    # category 5: physics
    directory_path = os.path.join(base_path, "physics")
    file_list = find_txt_files(directory_path)

    names_physics = []
    file_contents_physics = []
    labels_physics = []
    for file in file_list:
        names_physics.append(file.split("\\")[-1])
        file_contents_physics.append(read_txt_file(file))
        labels_physics.append("physics")

    # category 6: geography
    directory_path = os.path.join(base_path, "geography")
    file_list = find_txt_files(directory_path)

    names_geography = []
    file_contents_geography = []
    labels_geography = []
    for file in file_list:
        names_geography.append(file.split("\\")[-1])
        file_contents_geography.append(read_txt_file(file))
        labels_geography.append("geography")

    # category 7: biology
    directory_path = os.path.join(base_path, "biology")
    file_list = find_txt_files(directory_path)

    names_biology = []
    file_contents_biology = []
    labels_biology = []
    for file in file_list:
        names_biology.append(file.split("\\")[-1])
        file_contents_biology.append(read_txt_file(file))
        labels_biology.append("biology")

    names = (names_ComputerScience + names_History + names_Maths + names_accounts + names_physics + names_geography +
             names_biology)
    Y = (labels_ComputerScience + labels_History + labels_Maths + labels_accounts + labels_physics + labels_geography +
         labels_biology)

    # Create a mapping between unique strings and integers
    string_to_int = {string: index for index, string in enumerate(sorted(set(Y)))}

    # Convert the list of strings to a list of integers using the mapping
    Y_int_list = [string_to_int[string] for string in Y]

    data = (file_contents_ComputerScience + file_contents_History + file_contents_Maths + file_contents_accounts +
            file_contents_physics + file_contents_geography + file_contents_biology)

    return data, names, string_to_int, Y_int_list


def preprocess_data(data):
    data_words = list(sent_to_words(data))

    # Remove Stop Words
    print("Start removing stop words")
    data_words_nostops = remove_stopwords(data_words)

    # Do lemmatization keeping only noun, adj, vb, adv
    print("Start lemmatizing words")
    data_lemmatized = lemmatization(data_words_nostops, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    data_lemmatized_min_length = []

    for sublist in tqdm(data_lemmatized):
        # Use a list comprehension to filter out strings with less than two characters
        sublist = [word for word in sublist if len(word) > 2]
        data_lemmatized_min_length.append(sublist)

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized_min_length)

    # Create Corpus
    texts = data_lemmatized_min_length

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    return corpus, id2word


def corpus_to_sparse_dataframe(corpus, id2word):
    word_freq = dok_matrix((len(corpus), len(id2word)), dtype=int)

    for i, doc in enumerate(corpus):
        for word_id, freq in doc:
            word_freq[i, word_id] = freq

    dataframe = pd.DataFrame.sparse.from_spmatrix(word_freq)
    dataframe.columns = [id2word[word_id] for word_id in range(len(id2word))]
    return dataframe


def pairwise_distance_matrix(point, distance_function="euclidean"):
    """
	Compute the pairwise distance matrix of the point list
	You can use any distance function from scipy.spatial.distance.cdist or specify a callable function
	INPUT:
		ndarray: point: list of points
		str or callable: distance_function: distance function to use
	OUTPUT:
		ndarry: pairwise distance matrix
	"""
    if callable(distance_function):
        distance_matrix = cdist(point, point, distance_function)
    else:
        distance_matrix = cdist(point, point, distance_function)
    return distance_matrix


def knn_with_ranking(points, k, distance_function='euclidean'):
    """
    Compute the k-nearest neighbors of the points along with the
    rankings of other points based on the distance to each point.
    If the distance matrix is not provided, it is computed in O(n^2) time.
    INPUT:
    	ndarray: points: list of points
        int: k: number of nearest neighbors to compute
    	ndarray: distance_matrix: pairwise distance matrix (Optional)
    OUTPUT:
    	ndarray: knn_indices: k-nearest neighbors of each point
    	ndarray: ranking: ranking of other points based on the distance to each point
    """
    distance_matrix = pairwise_distance_matrix(points, distance_function)

    knn_indices = np.empty((points.shape[0], k), dtype=np.int32)
    ranking = np.empty((points.shape[0], points.shape[0]), dtype=np.int32)

    for i in range(points.shape[0]):
        distance_to_i = distance_matrix[i]
        sorted_indices = np.argsort(distance_to_i)
        knn_indices[i] = sorted_indices[1:k + 1]
        ranking[i] = np.argsort(sorted_indices)

    return knn_indices, ranking


def knn(points, k, distance_function="euclidean"):
    """
    Compute the k-nearest neighbors of the points
    You can use any distance function supported by scikit-learn KD Tree or specify a callable function
    INPUT:
    	ndarray: points: list of points
    	int: k: number of nearest neighbors to compute
    	str or callable: distance_function: distance function to use
    OUTPUT:
    	ndarray: knn_indices: k-nearest neighbors of each point
    """

    neigh = NearestNeighbors(n_neighbors=k, metric=distance_function)
    neigh.fit(points)

    knn_indices = neigh.kneighbors(points, k, return_distance=False)

    return knn_indices


def tnc_measure(orig, emb, k=20, return_local=False):
    """
    Compute the trustworthiness and continuity of the embedding
    INPUT:
        ndarray: orig: original data
        ndarray: emb: embedded data
        int: k: number of nearest neighbors to consider
        tuple: knn_ranking_info: precomputed k-nearest neighbors and rankings of the original and embedded data (Optional)
    OUTPUT:
        dict: trustworthiness and continuity
    """

    orig_knn_indices, orig_ranking = knn_with_ranking(orig, k, distance_function='cosine')
    emb_knn_indices, emb_ranking = knn_with_ranking(emb, k)

    if return_local:
        trust, local_trust = tnc_computation(orig_knn_indices, orig_ranking, emb_knn_indices, k, return_local)
        cont, local_cont = tnc_computation(emb_knn_indices, emb_ranking, orig_knn_indices, k, return_local)
        return ({
                    "trustworthiness": trust,
                    "continuity": cont
                }, {
                    "local_trustworthiness": local_trust,
                    "local_continuity": local_cont
                })
    else:
        trust = tnc_computation(orig_knn_indices, orig_ranking, emb_knn_indices, k, return_local)
        cont = tnc_computation(emb_knn_indices, emb_ranking, orig_knn_indices, k, return_local)
        return {
            "trustworthiness": trust,
            "continuity": cont
        }


def tnc_computation(base_knn_indices, base_ranking, target_knn_indices, k, return_local=False):
    """
    Core computation of trustworthiness and continuity
    """
    local_distortion_list = []
    points_num = base_knn_indices.shape[0]

    for i in range(points_num):
        missings = np.setdiff1d(target_knn_indices[i], base_knn_indices[i])
        local_distortion = 0.0
        for missing in missings:
            local_distortion += base_ranking[i, missing] - k
        local_distortion_list.append(local_distortion)
    local_distortion_list = np.array(local_distortion_list)
    local_distortion_list = 1 - local_distortion_list * (2 / (k * (2 * points_num - 3 * k - 1)))

    average_distortion = np.mean(local_distortion_list)

    if return_local:
        return average_distortion, local_distortion_list
    else:
        return average_distortion


def mrre_measure(orig, emb, k=20, return_local=False):
    """
    Compute Mean Relative Rank Error (MRRE) of the embedding
    INPUT:
        ndarray: orig: original data
        ndarray: emb: embedded data
        int: k: number of nearest neighbors to consider
        tuple: knn_ranking_info: precomputed k-nearest neighbors and rankings of the original and embedded data (Optional)
    OUTPUT:
        dict: MRRE_false and MRRE_missing
    """
    orig_knn_indices, orig_ranking = knn_with_ranking(orig, k, distance_function='cosine')
    emb_knn_indices, emb_ranking = knn_with_ranking(emb, k)

    if return_local:
        mrre_false, local_mrre_false = mrre_computation(orig_ranking, emb_ranking, emb_knn_indices, k, return_local)
        mrre_missing, local_mrre_missing = mrre_computation(emb_ranking, orig_ranking, orig_knn_indices, k,
                                                            return_local)
        return ({
                    "mrre_false": mrre_false,
                    "mrre_missing": mrre_missing
                }, {
                    "local_mrre_false": local_mrre_false,
                    "local_mrre_missing": local_mrre_missing
                })
    else:
        mrre_false = mrre_computation(orig_ranking, emb_ranking, emb_knn_indices, k, return_local)
        mrre_missing = mrre_computation(emb_ranking, orig_ranking, orig_knn_indices, k, return_local)

        return {
            "mrre_false": mrre_false,
            "mrre_missing": mrre_missing,
        }


def mrre_computation(base_ranking, target_ranking, target_knn_indices, k, return_local=False):
    """
    Core computation of MRRE
    """
    local_distortion_list = []
    points_num = target_knn_indices.shape[0]
    for i in range(points_num):
        base_rank_arr = base_ranking[i][target_knn_indices[i]]
        target_rank_arr = target_ranking[i][target_knn_indices[i]]
        local_distortion_list.append(np.sum(np.abs(base_rank_arr - target_rank_arr) / target_rank_arr))

    c = sum([abs(points_num - 2 * i + 1) / i for i in range(1, k + 1)])
    local_distortion_list = np.array(local_distortion_list)
    local_distortion_list = 1 - local_distortion_list / c

    average_distortion = np.mean(local_distortion_list)

    if return_local:
        return average_distortion, local_distortion_list
    else:
        return average_distortion


def lcmc_measure(orig, emb, k=7, return_local=False):
    """
    Compute the local continuity meta-criteria of the embedding
    INPUT:
    	ndarray: orig: original data
    	ndarray: emb: embedded data
    	int: k: number of nearest neighbors to consider
    	tuple: knn_info: precomputed k-nearest neighbors and rankings of the original and embedded data (Optional)
    OUTPUT:
    	dict: local continuity meta-criteria
    """

    orig_knn_indices = knn(orig, k, distance_function='cosine')
    emb_knn_indices = knn(emb, k)

    point_num = orig.shape[0]
    local_distortion_list = []

    for i in range(point_num):
        local_distortion_list.append(
            len(np.intersect1d(orig_knn_indices[i], emb_knn_indices[i])) / k)

    local_distortion_list = np.array(local_distortion_list)
    average_distortion = np.mean(local_distortion_list)

    if return_local:
        return ({
                    "lcmc": average_distortion
                }, {
                    "local_lcmc": local_distortion_list
                })
    else:
        return {
            "lcmc": average_distortion
        }


def nbh_measure(emb, label, k=7, return_local=False):
    """
    Compute neighborhood hit of the embedding
    INPUT:
    	ndarray: emb: embedded data
    	ndarray: label: label of the original data
    	int: k: number of nearest neighbors to consider
    	tuple: knn_info: precomputed k-nearest neighbors of the original and embedded data (Optional)
    OUTPUT:
    	dict: neighborhood hit (nh)
    """
    label = np.array(label)
    emb_knn_indices = knn(emb, k)

    points_num = emb.shape[0]
    nh_list = []
    for i in range(points_num):
        emb_knn_index = emb_knn_indices[i]
        emb_knn_index_label = label[emb_knn_index]
        nh_list.append(np.sum((emb_knn_index_label == label[i]).astype(int)))

    nh_list = np.array(nh_list)
    nh_list = nh_list / k

    nh = np.mean(nh_list)

    if return_local:
        return ({
                    "neighborhood_hit": nh
                }, {
                    "local_neighborhood_hit": nh_list
                })
    else:
        return {
            "neighborhood_hit": nh
        }


def get_mnist_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    X = np.zeros((x_train.shape[0], 784))

    for i in range(x_train.shape[0]):
        X[i] = x_train[i].flatten()

    y = y_train
    names = list(range(len(X)))
    print("Got MNIST dataset")

    return X, y, names


def main():
    # high_data, y, names = get_mnist_dataset()
    # tsne_results = get_tsne_layout(high_data, y, learning_rate=250, savefig_name="plot_mnist.png")
    # get_metrics(high_data, names, tsne_results, y, "results_mnist.csv")

    high_data, document_topic_matrix, names, y = get_seven_categories_dataset()
    tsne_results = get_tsne_layout(document_topic_matrix, y, metric=jensenshannon,
                                   savefig_name="plot_seven_categories_learning_rate_auto.png")
    get_metrics(high_data, names, tsne_results, y, "results_seven_categories_learning_rate_auto.csv")


def get_tsne_layout(x, y, savefig_name="t_SNE",
                    n_iter=1000, perplexity=30, learning_rate="auto", metric="euclidean"):
    x = np.array(x)
    print("Begin creating t-SNE layout")
    time_start = time.time()
    tsne = TSNE(n_components=2, n_iter=n_iter, perplexity=perplexity, learning_rate=learning_rate, metric=metric)
    tsne_results = tsne.fit_transform(x)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

    # Create the figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, title='TSNE')
    # Create the scatter
    ax.scatter(
        x=tsne_results[:, 0],
        y=tsne_results[:, 1],
        c=y,
        cmap=plt.cm.get_cmap('Paired'),
        alpha=0.4)
    plt.savefig(savefig_name)
    plt.close()
    return tsne_results


def get_metrics(high_data, names, tsne_results, y, res_file_name):
    tnc = tnc_measure(high_data, tsne_results, k=7, return_local=True)
    print("Got TNC measurements")
    trustworthiness_list = tnc[1]['local_trustworthiness'].tolist()
    continuity_list = tnc[1]['local_continuity'].tolist()
    mrre = mrre_measure(high_data, tsne_results, k=7, return_local=True)
    print("Got MRRE measurements")
    mrre_false_list = mrre[1]['local_mrre_false'].tolist()
    mrre_missing_list = mrre[1]['local_mrre_missing'].tolist()
    lcmc = lcmc_measure(high_data, tsne_results, k=7, return_local=True)
    print("Got LCMC measurements")
    lcmc_list = lcmc[1]['local_lcmc'].tolist()
    nbh = nbh_measure(tsne_results, y, k=7, return_local=True)
    print("Got neighborhood measurement")
    nh_list = nbh[1]["local_neighborhood_hit"].tolist()
    x = [tsne_point[0] for tsne_point in tsne_results]
    y = [tsne_point[1] for tsne_point in tsne_results]
    headers = ["Names", "x", "y", "Trustworthiness", "Continuity", "MRRE_False", "MRRE_Missing",
               "LCMC", "Neighborhood_Hit"]
    table = np.transpose(np.array([names, x, y, trustworthiness_list, continuity_list, mrre_false_list,
                                   mrre_missing_list, lcmc_list, nh_list]))
    df = pd.DataFrame(table, columns=headers)
    df.to_csv(res_file_name)


def get_seven_categories_dataset():
    data, names, mapping, y = gather_data("seven_categories_data")
    corpus, id2word = preprocess_data(data)
    VSM = corpus_to_sparse_dataframe(corpus, id2word)
    print("Got VSM model")
    K = 7
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=K,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=400,
                                                passes=30,
                                                alpha='auto',
                                                per_word_topics=True)
    print("Got LDA model")
    rows = []
    for doc in corpus:
        doc_top = []
        for t in lda_model.get_document_topics(doc, minimum_probability=0):
            doc_top.append(t[1])
        rows.append(doc_top)
    print("Created document_topic_matrix")
    return VSM.values, rows, names, y


if __name__ == "__main__":
    main()
