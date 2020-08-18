import numpy as np
import matplotlib.pyplot as plt
import torch as torch

from distill import *
from extract_data import *
from model import *

def main():
    np.random.seed(seed=2398)

    datasets_new = {'cifar' : {'data' : 'dataset/cifar10.npy', 'label' : 'label/cifar10.npy'}, 
                'mnist' : {'data' : 'dataset/mnist.npy', 'label' : 'label/mnist.npy'}}

    b1 = unpickle("../cifar-10/data_batch_1")
    data = b1[b'data']
    data.resize((10000, 3, 32, 32))
    distilled_data = distill(data)
    flattened_data = distilled_data.flatten()
    # with open('cifar10.npy', 'wb') as f:
    #     np.save(f, flattened_data)
    
    
    mnist_train = read_idx("../mnist/mnist-train")
    mnist_test = read_idx("../mnist/mnist-test")
    fashion_mnist_train = read_idx("../fashion-mnist/fashion-mnist-train")
    fashion_mnist_test = read_idx("../fashion-mnist/fashion-mnist-test")
    print(mnist_train.shape)
    print(mnist_test.shape)
    print(fashion_mnist_train.shape)
    print(fashion_mnist_test.shape)
    distilled_fashion_mnist = distill(fashion_mnist_test).flatten()
    with open('dataset/fashion_mnist.npy', 'wb') as f:
        np.save(f, distilled_fashion_mnist)

    

    # flattened_data = flattened_data.reshape((1, 216))
    # labs = np.zeros((1, 5))
    # labs[0, 2] = 1
    # dataset = organize_data(flattened_data, labs)
    # model, optimizer = init_model(216, 100, 100, 5, 0.001)
    # # train_model(dataset, model, optimizer, 20)
    # # with open('model.pb', 'wb') as f:
    # #     torch.save(model, f)

    # with open('model.pb', 'rb') as f:
    #     model = torch.load(f)
    # model.eval()
    # x = torch.from_numpy(flattened_data)
    # x = x.to(dtype=torch.float, device=device)
    # print(model(x))
    
    #read_images("../caltech101/101_ObjectCategories", "jpg")

if __name__ == '__main__':
    main()
