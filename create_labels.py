import numpy as np

file_name = 'label/blood_cell_images.npy'

#Scaling, Augmentation, Normalization, Centering, Remove Background, Object detection, Denoise, Perturb images, contrast, Dimensionality, Histogram, label one hot encoding
with open(file_name, 'wb') as f:
    #[0,0,0,0,0,0,0,0,0,0,0,0]
    arr = np.array([3,3,3,0,0,0,0,0,0,0,0,0])
    np.save(f, arr)
    f.close()