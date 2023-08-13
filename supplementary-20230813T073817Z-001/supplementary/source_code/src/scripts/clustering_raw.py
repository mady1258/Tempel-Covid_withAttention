from src.utils import utils
from src.data import cluster
from src.visualization import visualize
from src.data import make_dataset
import pandas as pd

# data_files = ['2011.csv', '2012.csv', '2013.csv', '2014.csv', '2015.csv', '2016.csv']
data_files = ['2013.csv', '2014.csv', '2015.csv', '2016.csv']
data_files = ['2016.csv']
data_path = './data/raw/'

# clustering_methods = ['DBSCAN']
reduction_method = 'TSNE'
reduction_method = 'PCA'
visualized_dimensions = 3 
method = 'DBSCAN'

# raw data for clustering
strains_by_year = make_dataset.read_strains_from(data_files, data_path) 
# processed data for clustering
trigram_vecs, _, _, _ = utils.read_and_process_to_trigram_vecs(data_files, data_path, sample_size=0, squeeze=False)

strains_by_year = cluster.label_encode(strains_by_year)
prot_vecs = cluster.squeeze_to_prot_vecs(trigram_vecs)

print(f'Shape: {len(strains_by_year)}x{len(strains_by_year[0])}x{len(strains_by_year[0][0])}')

clusters_by_year = cluster.cluster_raw(strains_by_year, prot_vecs)
# print(f'Number of clusters in the first year: {len(clusters_by_year[0]["centroids"])}')
# average = cluster.evaluate_clusters(clusters_by_year)
# print(f'Average variance of {method}: {average}')

# clusters_by_year = cluster.link_clusters(clusters_by_year)
clusters_by_year = cluster.remove_outliers(clusters_by_year)
visualize.show_clusters(clusters_by_year, method=reduction_method, dims=visualized_dimensions)