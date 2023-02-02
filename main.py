import numpy as np
import pandas as pd
import tensorly as tl

import compress
import loader

tl.set_backend('numpy')


def main():
    load = loader.Loader('C:/Users/SAM/EPIC-KITCHENS',
                         'C:/Users/SAM/Documents/GitHub/epic-kitchens-100-annotations')
    train_X, train_Y = load.get_train(8)
    print(train_X.shape)


if __name__ == '__main__':
    main()
