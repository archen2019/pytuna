import numpy as np
import torch as torch

from model import processed_data_and_labels_dict, organize_data, init_model, train_model

def main():
    data = np.zeros((1, 30, 224, 224), dtype='float32')
    labels = np.zeros((1, 12), dtype='int')
    for i in processed_data_and_labels_dict.keys():
        print(i)
        data_arr = np.load('rgbstack' + processed_data_and_labels_dict[i]['data'])
        data = np.append(data, data_arr, axis=0)
        labels_arr = np.load('01' + processed_data_and_labels_dict[i]['label'])
        labels = np.append(labels, np.tile(labels_arr, (40, 1)), axis=0)

    data = data[1:]
    labels = labels[1:]
    print("finished loading")
    # shuffled_data = np.zeros(data.shape, dtype=data.dtype)
    # shuffled_labels = np.zeros(labels.shape, dtype=labels.dtype)

    # permutation = np.random.permutation(len(data))
    # for old_index, new_index in enumerate(permutation):
    #     shuffled_data[new_index] = data[old_index]
    #     shuffled_labels[new_index] = labels[old_index]

    dataset = organize_data(data, labels)
    #model, optimizer = init_model(216, 1024, 400, 12, 1e-3)
    model, optimizer = init_model(1e-3, 12)
    train_model(dataset, model, optimizer, epochs=10)

    with open('model.pb', 'wb') as f:
        torch.save(model, f)

if __name__ == '__main__':
    main()
