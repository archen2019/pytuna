from os.path import isdir
from ast import literal_eval
from torchvision import transforms
import torch as torch

from .utils import read_images, distill

ALLOWED_IMAGE_FILE_TYPES = ['jpeg', 'jpg', 'png', 'bmp']
KNOWN_MODEL_DICT = {'vgg' : (224, 224), 'resnet' : (224, 224)}


def main():
    dataset_dir = input("Welcome to Preprocess.AI! This short wizard will guide you through preprocessing your image data. For each field, you may press enter to fill it with the provided default value.\nFirst, we need the path to the directory containing the image dataset you are classifying (default: \"dataset\"): ")
    while(not isdir(dataset_dir)):
        dataset_dir = input("It doesn't look like that's a directory. You can input a path relative to the directory from which you are running this module or an absolute path (default: \"dataset\"): ")

    user_dataset = []
    for filetype in ALLOWED_IMAGE_FILE_TYPES:
        user_dataset += read_images(dataset_dir, filetype)
    if not len(user_dataset):
        print("It doesn't seem like you have any images in the selected directory. Please restart the module with a different directory")
        return
    
    model_file = input("Now, let's load the model which will predict what preprocessing steps to use. By default, we'll just use the model which comes prepackaged with this module (default: model.pb): ") or "model.pb"
    model = None
    with open(model_file, 'rb') as f:
        model = torch.load(f)
    model.eval()
    X = torch.from_numpy(distill(user_dataset, 200).reshape((1, 216))).to(dtype=torch.float, device=torch.device('cuda'))
    output = model(X)
    print(output)

    transfer_learning = input("Do you intend to apply transfer learning to a pre-trained model? (default: no): ") or "no"
    if transfer_learning.lower() == "yes":
        transfer_model = input("Which model are you using? If the model doesn't appear in this list {}, then type other (default: other): ".format("(" + " ".join(KNOWN_MODEL_DICT.keys()) + ")")) or "other"
        if transfer_model in KNOWN_MODEL_DICT.keys():
            input_dim = KNOWN_MODEL_DICT[transfer_model]
        else:
            input_dim = literal_eval(input("What are the input dimensions of your model? (default: (224, 224)): ") or "(224, 224)")
        resize_transform = transforms.Compose([transforms.Resize(input_dim)])
        for i in range(len(user_dataset)):
            user_dataset[i] = resize_transform(user_dataset[i])
    else:
        dim_mismatch = False
        for i in range(len(user_dataset) - 1):
            if user_dataset[i].size != user_dataset[i + 1].size:
                dim_mismatch = True
                break
        if dim_mismatch:
            resize_dim = literal_eval(input("It seems like the images in your dataset have different sizes. Most models will only work with images of a fixed dimension, so we highly suggest resizing all your images to one size. What size would you like to resize to? Input (0, 0) if you wouldn't like to resize (default: (224, 224)): ") or "(224, 224)")
            if resize_dim != (0, 0):
                resize_transform = transforms.Compose([transforms.Resize(resize_dim)])
                for i in range(len(user_dataset)):
                    user_dataset[i] = resize_transform(user_dataset[i])

    tensor_compose = transforms.Compose([transforms.ToTensor()])
    for i in range(len(user_dataset)):
        user_dataset[i] = tensor_compose(user_dataset[i])

    zero_centering = input("Now that your images are resized, would you like to zero-center them? This will set the mean to 0 and standard deviation to 1 across each channel. We highly suggest this to remove artifacts like a blue tint on every image. (default: yes): ") or "yes"
    if zero_centering.lower() != "no":
        norm_mean = (0.49139968, 0.48215827, 0.44653124)
        norm_std = (0.24703233, 0.24348505, 0.26158768)
        normalization_transform = transforms.Compose([transforms.Normalize(norm_mean, norm_std)])
        for i in range(len(user_dataset)):
            user_dataset[i] = normalization_transform(user_dataset[i])

if __name__=='__main__':
    main()
