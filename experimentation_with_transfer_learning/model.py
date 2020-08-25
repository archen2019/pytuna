import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import numpy as np
import os

processed_data_and_labels_dict = {'cifar10' : {'data' : 'data/cifar10.npy', 'label' : 'label/cifar_10.npy'}, 
                'bees': {'data' : 'data/bees.npy', 'label': 'label/the_bee_image_dataset.npy'},
                'birds': {'data' : 'data/birds.npy', 'label' : 'label/caltech_ucsd.npy'},
                'bloodcells' : {'data' : 'data/bloodcells.npy', 'label' : 'label/blood_cell_images.npy'},
                'breastcancer' : {'data' : 'data/breastcancer.npy', 'label' : 'label/breast_histopathology_dataset.npy'},
                'caltech101': {'data' : 'data/caltech101.npy', 'label': 'label/caltech_101.npy'},
                'caltech256' : {'data' : 'data/caltech256.npy', 'label' : 'label/caltech_256.npy'},
                'captcha' : {'data' : 'data/captcha.npy', 'label' : 'label/captcha_images.npy'},
                'cars' : {'data' : 'data/cars.npy', 'label' : 'label/stanford_cars_dataset.npy'},
                'celebs' : {'data' : 'data/celeb.npy', 'label' : 'label/celebs.npy'},
                'coil20' : {'data' : 'data/coil20.npy', 'label' : 'label/coil_20.npy'},
                'coil100' : {'data' : 'data/coil100.npy', 'label' : 'label/coil_100.npy'},
                'cityscape' : {'data' : 'data/cityscape.npy', 'label' : 'label/cityscapes_image_pairs.npy'},
                'dogs' : {'data' : 'data/dogs.npy', 'label' : 'label/stanford_dogs_dataset.npy'},
                'fashion_mnist' : {'data' : 'data/fashion.npy', 'label' : 'label/fashion_mnist.npy'},
                'feret' : {'data' : 'data/feret.npy', 'label' : 'label/feret.npy'},
                'flowers' : {'data' : 'data/flowers.npy', 'label' : 'label/flowers.npy'},
                'flickr' : {'data' : 'data/flickr.npy', 'label' : 'label/flickr_image_dataset.npy'},
                'food41' : {'data' : 'data/food41.npy', 'label': 'label/food_images.npy'},
                'fruits' : {'data' : 'data/fruits.npy', 'label': 'label/fruits.npy'},
                'horsehuman' : {'data' : 'data/horsehuman.npy', 'label' : 'label/horses_or_humans.npy'},
                'imagenet' : {'data' : 'data/imagenet.npy', 'label': 'label/imagenet.npy'},
                'intel' : {'data' : 'data/intel.npy', 'label': 'label/intel_images_dataset.npy'},
                'komatsuna' : {'data' : 'data/komatsuna.npy', 'label': 'label/komatsuna.npy'},
                'lipreading' : {'data' : 'data/lipreading.npy', 'label': 'label/lip_reading_image_dataset.npy',
                'lsun' : {'data' : 'data/lsun.npy', 'label': 'label/lsun.npy'}},
                'malaria' : {'data' : 'data/malaria.npy', 'label': 'label/malaria_cell.npy'},
                'mnist' : {'data' : 'data/mnist.npy', 'label' : 'label/mnist.npy'},
                'monkey' : {'data' : 'data/monkey.npy', 'label' : 'label/10_monkey_species.npy'},
                'mscoco' : {'data' : 'data/mscoco.npy', 'label': 'label/ms_coco.npy'},
                'nist' : {'data' : 'data/nist.npy', 'label': 'label/nist.npy'},
                'openimg' : {'data' : 'data/openimg.npy', 'label': 'label/open_images_dataset.npy'},
                'overhead' : {'data' : 'data/overhead.npy', 'label': 'label/overhead_imagery_research_dataset.npy'},
                'pascalvoc' : {'data' : 'data/pascalvoc.npy', 'label': 'label/pascal_voc.npy'},
                'pet' : {'data' : 'data/pet.npy', 'label': 'label/oxford_iiit_pet_dataset.npy'},
                'pneumonia' : {'data' : 'data/pneumonia.npy', 'label' : 'label/pneumonia.npy'},
                'skincancer': {'data' : 'data/skincancer.npy', 'label' : 'label/skin_cancer.npy'},
                'visualqa': {'data' : 'data/visualqa.npy', 'label' : 'label/visual_qa.npy'},
                'worms' : {'data' : 'data/worms.npy', 'label' : 'label/broad_bioimage_benchmark_collection.npy'}
                }

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# def init_model(input_size, h1, h2, output_size, learning_rate):
#     model = nn.Sequential(
#         nn.Linear(input_size, h1),
#         nn.ReLU(),
#         nn.Linear(h1, h1),
#         nn.ReLU(),
#         nn.Linear(h1, h2),
#         nn.ReLU(),
#         nn.Linear(h2, output_size),
#         nn.ReLU(),
#         nn.Linear(output_size, output_size),
#         #nn.ReLU(),
#         nn.Sigmoid()
#     )

#     optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
#     if torch.cuda.device_count() > 1:
#         print("Let's use", torch.cuda.device_count(), "GPUs!")
#         model = nn.DataParallel(model)

#     return model, optimizer

def init_model(learning_rate=1e-3, output_size=12, unfreeze=1):
    resnet = models.resnext101_32x8d(pretrained=True)
    resnet.fc = nn.Linear(resnet.fc.in_features, output_size)

    ct = 0
    for child in resnet.children():
        ct += 1
        if ct < unfreeze:
            for param in child.parameters():
                param.requires_grad = False

    model = nn.Sequential(
        nn.Conv2d(30, 3, 3, padding=1),
        nn.ReLU(),
        resnet
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    return model, optimizer 

def init_model3d(learning_rate=1e-3, output_size=12, unfreeze=1):
    resnet = models.resnext101_32x8d(pretrained=True)
    resnet.fc = nn.Linear(resnet.fc.in_features, output_size)

    ct = 0
    for child in resnet.children():
        ct += 1
        if ct < unfreeze:
            for param in child.parameters():
                param.requires_grad = False

    model = nn.Sequential(
        nn.Conv3d(10, 1, (1, 3, 3), padding=(0, 1, 1)),
        nn.ReLU(),
        nn.Flatten(1,2),
        resnet
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    return model, optimizer     

def train_model(dataset, model, optimizer, epochs=1):
    model = model.to(device=device)

    for e in range(epochs):
        for t, (x, y) in enumerate(dataset):
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            scores = model(x)
            loss = F.mse_loss(scores, y)
            #loss = nn.BCELoss()(scores, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if t % 5 == 0:
                print("Iteration %d, loss = %.4f" % (t, loss.item()))
                print()

def organize_data(inputs, labels):
    assert inputs.shape[0] == labels.shape[0]

    dataset = []
    for i in range(inputs.shape[0]):
        dataset.append((torch.reshape(torch.from_numpy(inputs[i]), (1, 30, 224, 224)), torch.reshape(torch.from_numpy(labels[i]), (1, 12))))

    return dataset

def organize_data3d(inputs, labels):
    assert inputs.shape[0] == labels.shape[0]

    dataset = []
    for i in range(inputs.shape[0]):
        dataset.append((torch.reshape(torch.from_numpy(inputs[i]), (1, 10, 3, 224, 224)), torch.reshape(torch.from_numpy(labels[i]), (1, 12))))

    return dataset
