import numpy as np
from PIL import Image
from skimage.transform import resize
from torchvision import transforms

def distill(data, num_samples=100, width=6, height=6):
    idxs = np.random.random_integers(data.shape[0], size=num_samples)
    sample = data[idxs]
    dsample = np.zeros((num_samples, 3, width, height))

    norm_mean = (0.49139968, 0.48215827, 0.44653124)
    norm_std = (0.24703233, 0.24348505, 0.26158768)

    process = transforms.Compose([transforms.Resize((width, height)), 
                                    transforms.ToTensor(), 
                                    transforms.Normalize(norm_mean, norm_std)])

    for i, img in enumerate(sample):
        pil_img = Image.fromarray(img, 'RGB')
        dpil_img = process(pil_img)
        dimg = np.asarray(dpil_img)
        dsample[i] = dimg

    distilled_data = np.zeros((2, 3, width, height))

    distilled_data[0] = np.mean(dsample, axis=0)
    distilled_data[1] = np.std(dsample, axis=0)

    return distilled_data

