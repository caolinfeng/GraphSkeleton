import numpy as np

def dataloader(dataset):

    if dataset == 'dgraph':
        root = '../datasets/DGraphFin'
        file_path = root + '/dgraphfin.npz'
        data = np.load(file_path)

    return data, root