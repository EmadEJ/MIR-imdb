import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import sys
sys.path.append(os.path.abspath(os.path.join('Logic', 'core')))

from word_embedding.fasttext_data_loader import FastTextDataLoader
from word_embedding.fasttext_model import FastText
from .dimension_reduction import DimensionReduction
from .clustering_metrics import ClusteringMetrics
from .clustering_utils import ClusteringUtils

# Main Function: Clustering Tasks
if __name__ == '__main__':
    # 0. Embedding Extraction
    ft_model = FastText(method='skipgram')
    ft_model.prepare(None, mode='load')

    print("Starting embedding")

    path = 'preprocessed_docs.json'
    ft_data_loader = FastTextDataLoader(path)
    df = ft_data_loader.read_data_to_df()[:100]
    synopses = df['synopsis'].to_list()
    genres = df['genres'].to_list()
    cur_genres = []
    embeddings = []
    labels = []
    for idx in tqdm(range(len(synopses)), "embedding synopses"):
        synopsis = synopses[idx]
        genre = genres[idx]
        if len(synopsis.split()) == 0 or len(genre) == 0:
            continue
        embedding = ft_model.get_doc_embedding(synopsis)
        embeddings.append(embedding)
        if genre not in cur_genres:
            cur_genres.append(genre)
        labels.append(cur_genres.index(genre))

    embeddings = np.array(embeddings)
    labels = np.array(labels)

    print("Embedding done!")


    # 1. Dimension Reduction
    dr = DimensionReduction(n_components=10)

    dr.wandb_plot_explained_variance_by_components(embeddings, "MIR-Project", "test run")

    dr.wandb_plot_2d_tsne(embeddings, "MIR-Project", "tsne")

    # # 2. Clustering
    cu = ClusteringUtils()

    ## K-Means Clustering
    ks = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    embeddings_2d = dr.convert_to_2d_tsne(embeddings)
    for n_clusters in ks:
        cu.visualize_kmeans_clustering_wandb(embeddings_2d, n_clusters, "MIR-Project", "test run")
    cu.plot_kmeans_cluster_scores(embeddings_2d, labels, ks, "MIR-Project", "test run")
    cu.visualize_elbow_method_wcss(embeddings_2d, ks, "MIR-Project", "test run")

    ## Hierarchical Clustering
    cu.wandb_plot_hierarchical_clustering_dendrogram(embeddings, "MIR-Project", "average", "test run")
    cu.wandb_plot_hierarchical_clustering_dendrogram(embeddings, "MIR-Project", "ward", "test run")
    cu.wandb_plot_hierarchical_clustering_dendrogram(embeddings, "MIR-Project", "complete", "test run")
    cu.wandb_plot_hierarchical_clustering_dendrogram(embeddings, "MIR-Project", "single", "test run")

    # 3. Evaluation
    cm = ClusteringMetrics()
    best_k = 13 # according to number of genres
    _, kmean_labels = cu.cluster_kmeans(embeddings_2d, best_k)
    print("############################################ K-means:")
    print("rand score:", cm.adjusted_rand_score(labels, kmean_labels))
    print("purity score:", cm.purity_score(labels, kmean_labels))
    print("silhouette score:", cm.silhouette_score(embeddings_2d, kmean_labels))

    average_labels = cu.cluster_hierarchical_average(embeddings_2d, best_k)
    print("############################################ average hierarchical:")
    print("rand score:", cm.adjusted_rand_score(labels, average_labels))
    print("purity score:", cm.purity_score(labels, average_labels))
    print("silhouette score:", cm.silhouette_score(embeddings_2d, average_labels))

    ward_labels = cu.cluster_hierarchical_ward(embeddings_2d, best_k)
    print("############################################ ward hierarchical:")
    print("rand score:", cm.adjusted_rand_score(labels, ward_labels))
    print("purity score:", cm.purity_score(labels, ward_labels))
    print("silhouette score:", cm.silhouette_score(embeddings_2d, ward_labels))

    complete_labels = cu.cluster_hierarchical_complete(embeddings_2d, best_k)
    print("############################################ complete hierarchical:")
    print("rand score:", cm.adjusted_rand_score(labels, complete_labels))
    print("purity score:", cm.purity_score(labels, complete_labels))
    print("silhouette score:", cm.silhouette_score(embeddings_2d, complete_labels))

    single_labels = cu.cluster_hierarchical_single(embeddings_2d, best_k)
    print("############################################ single hierarchical:")
    print("rand score:", cm.adjusted_rand_score(labels, single_labels))
    print("purity score:", cm.purity_score(labels, single_labels))
    print("silhouette score:", cm.silhouette_score(embeddings_2d, single_labels))


# ############################################ K-means:
# rand score: 0.14648511411853646
# confusion matrix:
# [[5 3 1 2 0 0 0 0 0 0 0 0 0]
#  [0 3 0 0 1 4 0 5 0 0 8 2 2]
#  [0 1 3 4 0 1 1 0 7 3 1 1 0]
#  [0 0 0 0 0 3 0 1 0 0 3 2 6]
#  [0 0 0 2 0 1 1 0 0 0 0 0 1]
#  [0 1 0 0 4 0 1 0 0 0 0 0 1]
#  [0 1 0 0 2 0 0 2 0 1 0 1 0]
#  [0 0 1 0 0 0 0 0 0 0 0 1 0]
#  [0 0 0 0 0 0 0 0 0 1 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0 0 2 0]
#  [0 0 0 0 0 0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0 0 0 0]]
# purity score: 0.3917525773195876
# silhouette score: 0.28979993
# ############################################ average hierarchical:
# rand score: 0.08572425981279864
# confusion matrix:
# [[0 0 0 0 2 0 5 0 3 0 0 1 0]
#  [6 1 2 3 0 1 0 0 2 4 5 0 1]
#  [2 0 0 3 4 0 0 6 0 0 2 3 2]
#  [5 2 2 0 0 3 0 0 0 0 3 0 0]
#  [1 1 0 2 0 1 0 0 0 0 0 0 0]
#  [1 1 0 1 0 0 0 0 1 0 0 0 3]
#  [1 0 1 1 0 0 0 0 1 2 0 0 1]
#  [0 0 1 0 0 0 0 0 0 0 0 1 0]
#  [1 0 0 0 0 0 0 0 0 0 0 0 0]
#  [0 0 2 0 0 0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0 0 0 0]]
# purity score: 0.3402061855670103
# silhouette score: 0.32069874
# ############################################ ward hierarchical:
# rand score: 0.10728488668411612
# confusion matrix:
# [[0 0 0 0 0 0 0 0 3 2 0 5 1]
#  [3 2 1 1 6 3 2 0 2 0 5 0 0]
#  [3 0 1 1 0 2 0 6 0 4 2 0 3]
#  [0 2 5 1 0 3 4 0 0 0 0 0 0]
#  [2 0 0 1 0 0 1 0 0 0 1 0 0]
#  [1 0 1 4 0 0 0 0 1 0 0 0 0]
#  [1 1 0 1 2 0 0 0 1 0 1 0 0]
#  [0 1 0 0 0 0 0 0 0 0 0 0 1]
#  [0 0 0 0 0 0 0 0 0 0 1 0 0]
#  [0 1 0 0 0 1 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0 0 0 0]]
# purity score: 0.3402061855670103
# silhouette score: 0.36401924
# ############################################ complete hierarchical:
# rand score: 0.07136489826135238
# confusion matrix:
# [[0 5 0 0 2 0 0 1 0 0 3 0 0]
#  [2 2 1 3 0 5 0 0 2 4 2 4 0]
#  [1 2 1 3 4 2 0 3 0 0 0 2 4]
#  [5 0 1 0 0 3 2 0 4 0 0 0 0]
#  [1 0 1 2 0 0 0 0 1 0 0 0 0]
#  [1 0 4 1 0 0 0 0 0 0 1 0 0]
#  [0 0 1 1 0 0 1 0 0 2 1 1 0]
#  [0 1 0 0 0 0 0 1 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0 0 1 0]
#  [0 1 0 0 0 1 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0 0 0 0]]
# purity score: 0.30927835051546393
# silhouette score: 0.33743447
# ############################################ single hierarchical:
# rand score: 0.10583370448611344
# confusion matrix:
# [[ 0  3  4  0  0  2  0  0  0  0  0  2  0]
#  [ 0 25  0  0  0  0  0  0  0  0  0  0  0]
#  [ 2  6  6  2  4  2  0  0  0  0  0  0  0]
#  [ 0 13  0  0  0  0  1  0  0  0  0  0  1]
#  [ 0  2  0  0  0  0  0  0  0  1  2  0  0]
#  [ 0  6  0  0  0  0  0  0  1  0  0  0  0]
#  [ 0  6  0  0  0  0  0  1  0  0  0  0  0]
#  [ 0  1  1  0  0  0  0  0  0  0  0  0  0]
#  [ 0  1  0  0  0  0  0  0  0  0  0  0  0]
#  [ 0  2  0  0  0  0  0  0  0  0  0  0  0]
#  [ 0  0  0  0  0  0  0  0  0  0  0  0  0]
#  [ 0  0  0  0  0  0  0  0  0  0  0  0  0]
#  [ 0  0  0  0  0  0  0  0  0  0  0  0  0]]
