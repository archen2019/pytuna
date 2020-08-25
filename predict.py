import numpy as np
import torch
import torchvision
from model import processed_data_and_labels_dict, organize_data

def eval_dataset(scores, label):
    scores = scores.to(device=torch.device("cpu"))
    current_dataset_error = 0
    size = scores.size()
    for i in range(size[1]):
        current_dataset_error += np.power(scores[0][i] - label[i], 2)

    return current_dataset_error

model_file = 'model_custom.pb'
#model_file = 'model3dBCE.pb'

model = torch.load(model_file)
device = torch.device("cuda")
model.to(device)
model.eval()

total_dataset_error = 0
data = np.zeros((1, 30, 224, 224), dtype='float32')
label = np.zeros((1, 12), dtype='int')

for i in processed_data_and_labels_dict.keys():
    print(i)
    data_arr = np.load('rgbstack' + processed_data_and_labels_dict[i]['data'])
    data = np.append(data, data_arr, axis=0)
    labels_arr = np.load('01' + processed_data_and_labels_dict[i]['label'])
    label = np.append(label, np.reshape(labels_arr, (1,12)), axis=0)

data = data[1:]
label = label[1:]

for j in range(data.shape[0]):
    dataset = (torch.reshape(torch.from_numpy(data[j]), (1, 30, 224, 224)))
    dataset = dataset.to(device, dtype=torch.float)
    with torch.no_grad():
        scores = model(dataset)
        total_dataset_error += eval_dataset(scores, label[j])

print(total_dataset_error)    


