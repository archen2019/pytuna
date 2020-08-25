import numpy as np
from PIL import Image
from skimage.transform import resize
from torchvision import transforms

def distill(data, num_samples=20, num_imgs=100, width=6, height=6):
    process = transforms.Compose([transforms.Resize((width, height)), 
                                    transforms.ToTensor()])

    distilled_data = np.zeros((num_samples, width * height * 6))

    for j in range(num_samples):
        idxs = np.random.random_integers(0, len(data) - 1, size=num_imgs)
        sample = []
        for idx in idxs:
            sample.append(data[idx])
    
        dimgs = np.zeros((num_imgs, 3, width, height))
    
        for i, img in enumerate(sample):
            dimgs[i] = np.asarray(process(img))
    
        dsample = np.zeros((2, 3, width, height))
    
        dsample[0] = np.mean(dimgs, axis=0)
        dsample[1] = np.std(dimgs, axis=0)

        distilled_data[j] = dsample.flatten()
    
    return distilled_data

def stackimg(data, num_samples=40, num_imgs=10):
    
    stacked_images = np.zeros((num_samples, 3 * num_imgs, 224, 224))

    process = transforms.Compose([transforms.Resize((224, 224)), 
                                    transforms.ToTensor()])

    for i in range(num_samples):
        idxs = np.random.random_integers(0, len(data) - 1, size=num_imgs)
        for j, idx in enumerate(idxs):
            stacked_images[i, 3*j:(3*j+3)] = np.asarray(process(data[idx]))

    return stacked_images
