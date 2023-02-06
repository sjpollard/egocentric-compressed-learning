import numpy as np
import pandas as pd
import tensorly as tl

import compress
import loader

tl.set_backend('pytorch')


def preprocess_epic(loader):
    train_X, train_Y = loader.get_train(8)
    loader.save_to_pt('train_X.pt', train_X)
    loader.save_to_pt('train_Y.pt', train_Y)


def main():
    windows_load = loader.Loader('C:/Users/SAM/EPIC-KITCHENS',
                                 'C:/Users/SAM/Documents/GitHub/epic-kitchens-100-annotations',
                                 'C:/Users/SAM/Documents/GitHub/egocentric-compressed-learning/data')
    """ linux_load = loader.Loader('/home/hiraeth/EPIC-KITCHENS',
                               '/home/hiraeth/Github/epic-kitchens-100-annotations',
                               '/home/hiraeth/Github/egocentric-compressed-learning') """
    print(windows_load.load_from_pt('train_Y.pt').size())


if __name__ == '__main__':
    main()
