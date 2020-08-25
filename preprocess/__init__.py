from os.path import isdir
from os import mkdir
from shutil import rmtree
from ast import literal_eval
import torch as torch
from torchvision import transforms
from torchvision.transforms.functional import adjust_contrast, to_pil_image, to_tensor, normalize, resize
from PIL.ImageOps import equalize
from PIL.ImageFilter import GaussianBlur

from .utils import read_images, stackimg

ALLOWED_IMAGE_FILE_TYPES = ['jpeg', 'jpg', 'png', 'bmp', 'tiff']
KNOWN_MODEL_DICT = {'vgg' : (224, 224), 'resnet' : (224, 224), 'alexnet' : (224, 224), 'squeezenet' : (224, 224),
                    'densenet' : (224, 224), 'inceptionv3' : (229, 229), 'googlenet': (224, 224),
                    'shufflenetv2' : (224, 224), 'mobilenetv2' : (224, 224), 'resnext' : (224, 224),
                    'wideresnet': (224, 224), 'mnasnet' : (224, 224)}
PREDICTION_THRESHOLD = 0.5

def preprocess_wizard():
    dataset_dir = input("Welcome to PyTuna!\nThis short wizard will guide you through preprocessing your image data. For each field, you may press enter to fill it with the provided default value (the capitalized option).\nFirst, we need the path to the directory containing the image dataset you are classifying (default: \"dataset\"): ") or "dataset"
    while(not isdir(dataset_dir)):
        dataset_dir = input("It doesn't look like that's a directory. You can input a path relative to the directory from which you are running this module or an absolute path (default: \"dataset\"): ") or "dataset"

    user_dataset = []
    print("loading...")
    for filetype in ALLOWED_IMAGE_FILE_TYPES:
        user_dataset += read_images(dataset_dir, filetype)
    if not len(user_dataset):
        print("It doesn't seem like you have any images in the selected directory. Please restart the module with a different directory")
        return
    model_file = input("\nNow, let's load the model which will predict what preprocessing steps to use. By default, we'll just use the model which comes prepackaged with this module (default: model.pb): ") or "model.pb"
    model = None
    device = None
    model_location = input("\nWhere do you want to load the model? The prepackaged model is quite large, so we recommend choosing cuda only if you have more than 11 GB of GPU ram. (default: cpu) ") or "cpu"
    if torch.cuda.is_available() and model_location == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("loading...")
    with open(model_file, 'rb') as f:
        model = torch.load(f, device)
    print("predicting...")
    model.eval()
    X = torch.from_numpy(stackimg(user_dataset, 5, 10)).to(dtype=torch.float, device=device)
    model_output = model(X)
    model_output = torch.mean(model_output, axis=0)

    print("\nOur model has predicted which preprocessing steps should be used. Now, let's walk you through each of its suggestions.\n")

    transfer_learning = input("\nDo you intend to apply transfer learning to a pre-trained model? (y/N): ") or "n"
    if transfer_learning.lower() == "y":
        transfer_model = input("\nWhich model are you using? If the model doesn't appear in this list {}, then type other (default: other): ".format("(" + " ".join(KNOWN_MODEL_DICT.keys()) + ")")) or "other"
        if transfer_model in KNOWN_MODEL_DICT.keys():
            input_dim = KNOWN_MODEL_DICT[transfer_model]
        else:
            input_dim = literal_eval(input("\nWhat are the input dimensions of your model? (default: (224, 224)): ") or "(224, 224)")
        print("processing...")
        for i in range(len(user_dataset)):
            user_dataset[i] = resize(user_dataset[i], input_dim)
    else:
        dim_mismatch = False
        for i in range(len(user_dataset) - 1):
            if user_dataset[i].size != user_dataset[i + 1].size:
                dim_mismatch = True
                break
        if dim_mismatch:
            resize_dim = literal_eval(input("\nIt seems like the images in your dataset have different sizes. Most models will only work with images of a fixed dimension, so we highly suggest resizing all your images to one size. What size would you like to resize to? Input (0, 0) if you wouldn't like to resize (default: (224, 224)): ") or "(224, 224)")
            if resize_dim != (0, 0):
                print("processing...")
                for i in range(len(user_dataset)):
                    user_dataset[i] = resize(user_dataset[i], resize_dim)

    for i in range(len(user_dataset)):
        user_dataset[i] = to_tensor(user_dataset[i])

    if model_output[2] > PREDICTION_THRESHOLD:
        zero_centering = input("\nNow that your images are resized, would you like to zero-center them? This will set the mean to 0 and standard deviation to 1 across each channel. This step could remove artifacts like a blue tint on every image. The model highly suggests this step. (Y/n) ") or "y"
    else:
        zero_centering = input("\nNow that your images are resized, would you like to zero-center them? This will set the mean to 0 and standard deviation to 1 across each channel. This step could remove artifacts like a blue tint on every image. The model does not suggest this step. (y/N) ") or "n"
    if zero_centering.lower() != "n":
        norm_mean = (0.49139968, 0.48215827, 0.44653124)
        norm_std = (0.24703233, 0.24348505, 0.26158768)
        print("processing...")
        for i in range(len(user_dataset)):
            user_dataset[i] = normalize(user_dataset[i], norm_mean, norm_std)

    if model_output[1] > PREDICTION_THRESHOLD:
        augmentation = input("\nWould you like to augment your data? This will create a copy of the entire dataset and apply random affine transforms to each image, which increases the size of the training data and lets your model learn more. Note that this will double the size of the dataset, and so you should append a copy of the list of label tensors to itself to accomodate the change. The model highly suggests this step. (Y/n) ") or "y"
    else:
        augmentation = input("\nWould you like to augment your data? This will create a copy of the entire dataset and apply random affine transforms to each image, which increases the size of the training data and lets your model learn more. Note that this will double the size of the dataset, and so you should append a copy of the list of label tensors to itself to accomodate the change. The model does not suggest this step. (y/N) ") or "n"
    if augmentation.lower() != "n":
        augmentation_transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomGrayscale(p=0.4), transforms.RandomHorizontalFlip(p=0.4), transforms.RandomVerticalFlip(p=0.4), transforms.RandomApply([transforms.RandomRotation(90)], p=0.4), transforms.ToTensor()])
        old_len = len(user_dataset)
        print("processing...")
        for i in range(old_len):
            user_dataset.append(augmentation_transform(user_dataset[i]))

    if model_output[8] > PREDICTION_THRESHOLD:
        contrast = input("\nWould you like to increase the contrast of the images in your dataset? This can lead to easier feature detection. The model highly suggests this step. (Y/n) ") or "y"
    else:
        contrast = input("\nWould you like to increase the contrast of the images in your dataset? This can lead to easier feature detection. The model does not suggest this step. (y/N) ") or "n"
    if contrast.lower() != "n":
        print("processing...")
        for i in range(len(user_dataset)):
            user_dataset[i] = adjust_contrast(user_dataset[i], 2)

    if model_output[10] > PREDICTION_THRESHOLD:
        histogram_equalization = input("\nWould you like to histogram equalize your images? This will create a uniform distribution of grayscale values in each image. The model highly suggests this step. (Y/n) ") or "y"
    else:
        histogram_equalization = input("\nWould you like to histogram equalize your images? This will create a uniform distribution of grayscale values in each image. The model does not suggest this step. (y/N) ") or "n"
    if histogram_equalization.lower() != "n":
        print("processing...")
        for i in range(len(user_dataset)):
            user_dataset[i] = to_tensor(equalize(to_pil_image(user_dataset[i])))
    
    if model_output[6] > PREDICTION_THRESHOLD:
        gaussian_blur = input("\nWould you like to apply a Gaussian blur to your dataset? This will soften the edges in your images. The model highly suggests this step. (Y/n) ") or "y"
    else:
        gaussian_blur = input("\nWould you like to apply a Gaussian blur to your dataset? This will soften the edges in your images. The model does not suggest this step. (y/N) ") or "n"        
    if gaussian_blur.lower() != "n":
        print("processing...")
        for i in range(len(user_dataset)):
            user_dataset[i] = to_tensor(to_pil_image(user_dataset[i]).filter(GaussianBlur))

    if model_output[9] > PREDICTION_THRESHOLD:
        grayscaling = input("\nWould you like to grayscale your dataset? This will reduce the dimensionality of your dataset and could lead to faster convergence. The model highly recommends this step. (Y/n) ") or "y"
    else:
        grayscaling = input("\nWould you like to grayscale your dataset? This will reduce the dimensionality of your dataset and could lead to faster convergence. The model does not recommend this step. (y/N) ") or "n"
    if grayscaling.lower() != "n":
        grayscale_transform = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(), transforms.ToTensor()])
        print("processing...")
        for i in range(len(user_dataset)):
            user_dataset[i] = grayscale_transform(user_dataset[i])
    
    if model_output[4] > PREDICTION_THRESHOLD:
        input("\nYou could also try to remove the backgrounds in your dataset. While we can't do this manually, this step could remove noise and let the model learn about only the objects in the foreground for easier classification. The model highly recommends this step. (Press Enter to continue)" )
    else:
        input("\nYou could also try to remove the backgrounds in your dataset. While we can't do this manually, this step could remove noise and let the model learn about only the objects in the foreground for easier classification. The model does not recommend this step. (Press Enter to continue)" )        

    if model_output[5] > PREDICTION_THRESHOLD:
        input("\nYou could also run an object detection algorithm on your dataset to make sure your model focuses on the objects contained in the images. Some popular options include YOLO and SSD. The model highly recommends this step. (Press Enter to continue) ")
    else:
        input("\nYou could also run an object detection algorithm on your dataset to make sure your model focuses on the objects contained in the images. Some popular options include YOLO and SSD. The model does not recommend this step. (Press Enter to continue) ")

    save_to_disk = input("\nWould you like to save your preprocessed dataset to disk? This will create a new directory and populate it with pickled torch tensors representing each image. They can be loaded with torch.load('filename.pt') for future usage. If you choose not to, this function will still return a list of preprocessed tensors, ready for consumption by a model. (Y/n) ") or "y"
    if save_to_disk.lower() != "n":
        output_dir = input("\nWhat output directory would you like to save your preprocessed tensors to? (default: preprocessed_dataset) ") or "preprocessed_dataset"
        rmtree(output_dir, ignore_errors=True)
        mkdir(output_dir)
        print("saving...")
        for i in range(len(user_dataset)):
            with open(output_dir + '/' + format(i, '07d') + '.pt', 'wb') as f:
                torch.save(user_dataset[i], f)
    #print(user_dataset)
    return user_dataset