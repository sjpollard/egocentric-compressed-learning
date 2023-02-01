import numpy as np
import pandas as pd
import tensorly as tl

import compress
import loader

tl.set_backend('numpy')


def main():
    measurements = [25, 45, 1]
    train_annotations = pd.read_csv('annotations\EPIC_100_train.csv')
    for annotation in train_annotations.loc[0:0, ['participant_id', 'video_id', 'start_frame', 'stop_frame', 'verb', 'noun']].values:
        X, Y = loader.get_annotation_snippets(annotation, 8)
        W1 = compress.random_gaussian_matrix((measurements[2], X.shape[3]))
        W2 = compress.random_gaussian_matrix((measurements[1], X.shape[2]))
        W3 = compress.random_gaussian_matrix((measurements[0], X.shape[1]))
        print(compress.compress_tensor(X, [W1, W2, W3], [3, 2, 1]))


if __name__ == '__main__':
    main()
