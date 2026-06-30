from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

def build_gmm(data):
    model = GaussianMixture(n_components=5, random_state=0).fit(data) 
    return model
