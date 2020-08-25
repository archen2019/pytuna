import numpy as np

from distill import *
from extract_data import *

def generate_train_data(rootdir="../datasets/"):
    dsets = {}
    #dsets["bees"] = read_images(rootdir + "honey-bee", 'png')
    #dsets["birds"] = read_images(rootdir + "caltech-ucsd-birds", 'jpg')
    #dsets["bloodcells"] = read_images(rootdir + "blood-cells", 'jpg')
    #dsets["breastcancer"] = read_images(rootdir + "breast_cancer_histopathology_dataset", 'png')
    #dsets["cars"] = read_images(rootdir + "stanford-cars-dataset", 'jpg')
    #dsets["coil100"] = read_images(rootdir + "coil-100", 'png')
    #dsets["coil20"] = read_images(rootdir + "coil-20", 'png')
    #dsets["captcha"] = read_images(rootdir + "captcha-version-2", 'png')
    #dsets["caltech101"] = read_images(rootdir + "caltech101", 'jpg')
    #dsets["caltech256"] = read_images(rootdir + "caltech256", 'jpg')
    #dsets["celeb"] = read_images(rootdir + "celeba-dataset/img_align_celeba", 'jpg')
    #dsets["cifar10"] = unpickle_mnist(rootdir + "cifar-10/data_batch_1")
    #dsets["cityscape"] = read_images(rootdir + "cityscapes-image-pairs/cityscapes_data", 'jpg')
    #dsets["dogs"] = read_images(rootdir + "stanford-dogs-dataset", 'jpg')
    dsets["fashion"] = read_idx(rootdir + "fashion-mnist/fashion-mnist-train")
    #dsets["feret"] = read_images(rootdir + "feret", 'png')
    #dsets["flickr"] = read_images(rootdir + "flickr-image-dataset", 'jpg')
    #dsets["flowers"] = read_images(rootdir + "flowers", 'jpg')
    #dsets["food41"] = read_images(rootdir + "food-41", 'jpg')
    #dsets["fruits"] = read_images(rootdir + "fruits", 'jpg')
    #dsets["horsehuman"] = read_images(rootdir + "horses-or-humans", 'png')
    #dsets["imagenet"] = read_images(rootdir + "imagenet", 'JPEG')
    #dsets["intel"] = read_images(rootdir + "intel-image-classification", 'jpg')
    #dsets["komatsuna"] = read_images(rootdir + "komatsuna", 'png')
    #dsets["lipreading"] = read_images(rootdir + "lip-reading-image-dataset", 'jpg')
    #dsets["lsun"] = read_images(rootdir + "lsun", 'jpg')
    #dsets["malaria"] = read_images(rootdir + "malaria-cell", 'png')
    dsets["mnist"] = read_idx(rootdir + "mnist/mnist-train")
    #dsets["monkey"] = read_images(rootdir + "10-monkey-species", 'jpg')
    #dsets["mscoco"] = read_images(rootdir + "ms-coco", 'jpg')
    #dsets["nist"] = read_images(rootdir + "nist", 'png')
    #dsets["openimg"] = read_images(rootdir + "open-images-dataset", 'jpg')
    #dsets["overhead"] = read_images(rootdir + "overhead-imagery-research", 'tiff')
    #dsets["pascalvoc"] = read_images(rootdir + "pascal-voc", 'jpg')
    #dsets["pet"] = read_images(rootdir + "oxford-iiit-pet", 'jpg')
    #dsets["pneumonia"] = read_images(rootdir + "chest-xray-pneumonia", 'jpeg')
    #dsets["skincancer"] = read_images(rootdir + "skin-cancer-mnist-ham10000", 'jpg')
    #dsets["visualqa"] = read_images(rootdir + "visualqa", 'png')
    #dsets["worms"] = read_images(rootdir + "bbbc010-v2-worm", 'tiff')

    for name in dsets:
        print(name)
        with open(name + ".npy", 'wb') as f:
            np.save(f, stackimg(dsets[name]))
