import numpy as np
from boxprint import bprint
import os

def dataloader(dataset):

    if dataset == 'dgraph':
        bprint("DGraph-Fin", width=20)
        root = '../datasets/DGraphFin'
        file_path = root + '/dgraphfin.npz'
        data = np.load(file_path)
        save_path = root + '/skeleton'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    return data, save_path