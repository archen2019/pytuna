import glob
from PIL import Image
import torch as torch
from torchvision import transforms
import numpy as np

device = None
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# dir is directory of images, filetype is the filetype of the images
def read_images(dir, filetype):
    imgs = []
    for filename in glob.iglob(dir + '/**/*.' + filetype, recursive=True):
        img = Image.open(filename).convert('RGB')
        imgs.append(img.copy())

    return imgs

def stackimg(data, num_samples=40, num_imgs=10):
    
    stacked_images = np.zeros((num_samples, num_imgs, 3, 224, 224))

    process = transforms.Compose([transforms.Resize((224, 224)), 
                                    transforms.ToTensor()])

    for i in range(num_samples):
        idxs = np.random.random_integers(0, len(data) - 1, size=num_imgs)
        for j, idx in enumerate(idxs):
            stacked_images[i, j] = np.asarray(process(data[idx]))

    return stacked_images
