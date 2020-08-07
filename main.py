import numpy as np
import matplotlib.pyplot as plt

from distill import *
from extract_data import *

def main():
    np.random.seed(seed=2398)

    b1 = unpickle("../cifar-10/data_batch_1")
    data = b1[b'data']
    data.resize((10000, 3, 32, 32))
    distilled_data = distill(data)

if __name__ == '__main__':
    main()
