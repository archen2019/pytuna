import glob
import idx2numpy
import numpy as np
import pickle
from PIL import Image, ImageFile
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True

def unpickle_mnist(file):
    print(file)
    imgs = []
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        ndimgs = dict[b'data']
        for img in ndimgs:
            img = img.reshape(3, 32, 32)
            img = np.swapaxes(img, 0, -1)
            img = Image.fromarray(np.uint8(img * 255)).convert('RGB')
            imgs.append(img)

    return imgs

def read_idx(file):
    print(file)
    ndimgs = idx2numpy.convert_from_file(file)
    imgs = []
    for i in range(ndimgs.shape[0]):
        img = Image.fromarray(np.uint8(ndimgs[i])).convert('RGB')
        imgs.append(img)

    return imgs

# dir is directory of images, filetype is the filetype of the images
def read_images(dir, filetype):
    print(dir)
    imgs = []
    for filename in glob.iglob(dir + '/**/*.' + filetype, recursive=True):
        img = Image.open(filename).convert('RGB')
        imgs.append(img.copy())

    return imgs
