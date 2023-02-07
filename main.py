import numpy as np
import pandas as pd
import tensorly as tl
import torchvision
import compress
import data
import torch

tl.set_backend('pytorch')


def preprocess_epic(loader):
    train_X, train_Y = loader.get_train(8)
    loader.save_to_pt('train_X.pt', train_X)
    loader.save_to_pt('train_Y.pt', train_Y)


def main():
    windows_loader = data.Loader('C:/Users/SAM/EPIC-KITCHENS',
                                 'C:/Users/SAM/Documents/GitHub/epic-kitchens-100-annotations',
                                 'C:/Users/SAM/Documents/GitHub/egocentric-compressed-learning/data')
    """ linux_loader = data.Loader('/home/hiraeth/EPIC-KITCHENS',
                               '/home/hiraeth/Github/epic-kitchens-100-annotations',
                               '/home/hiraeth/Github/egocentric-compressed-learning') """
    #preprocess_epic(windows_loader)
    train_X, train_Y = windows_loader.load_from_pt('train_X.pt'), windows_loader.load_from_pt('train_Y.pt')
    clip = train_X[0].float()
    torchvision.transforms.functional.to_pil_image(clip[3]).show()
    print(tl.context(clip.float()))
    W1 = compress.random_gaussian_matrix((100, 224))
    W2 = compress.random_gaussian_matrix((100, 224))
    compressed_clip = compress.compress_tensor(clip.float(), [W1, W2], [3, 2])
    #expanded_clip = compress.expand_tensor()
    torchvision.transforms.functional.to_pil_image(compressed_clip[3]).show()



if __name__ == '__main__':
    main()
