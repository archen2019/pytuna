import numpy as np

from model import processed_data_and_labels_dict

for i in processed_data_and_labels_dict.keys():
    label = np.load(processed_data_and_labels_dict[i]['label'])
    label_c = label[:]
    if np.sum(label_c):
        label_c = label_c / np.sum(label_c)
    with open('sum1' + processed_data_and_labels_dict[i]['label'], 'wb') as f:
        np.save(f, label_c)
    label_c = label[:]
    for x in range(label_c.shape[0]):
        if label_c[x]:
            label_c[x] = 1
    with open('01' + processed_data_and_labels_dict[i]['label'], 'wb') as f:
        np.save(f, label_c)