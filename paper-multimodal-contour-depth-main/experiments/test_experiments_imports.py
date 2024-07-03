
if __name__ == "__main__":
    
    print("Testing requirements.txt imports")    
    from time import time
    import pickle
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from skimage.color import rgb2gray
    from skimage.io import imread
    from skimage.transform import resize, EuclideanTransform, warp
    from sklearn.metrics import adjusted_rand_score
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    import seaborn as sns
    print(" - DONE")

    print("Testing custom imports")
    import sys
    sys.path.insert(0, ".")
    sys.path.insert(0, "..")
    from src import data
    from src import clustering
    from src import depth
    from src import competing
    print(" - DONE")
