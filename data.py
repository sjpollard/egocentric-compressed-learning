import numpy as np
import pandas as pd
import tensorly as tl
import torch
import os
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset

tl.set_backend('pytorch')

class CustomClipDataset(Dataset):
  def __init__(self, dataset):
    assert dataset[0].size(0) == dataset[1].size(0)
    self.x = dataset[0]
    self.y = dataset[1]

  def __getitem__(self, index):
    return self.x[index], self.y[index]

  def __len__(self):
    return self.x.size(0)

class Preprocessor:
    def __init__(self, dataset_path, annotations_path, data_path):
        self.dataset_path = dataset_path
        self.train_annotations = pd.read_csv(
            f'{annotations_path}/EPIC_100_train.csv')
        self.val_annotations = pd.read_csv(
            f'{annotations_path}/EPIC_100_validation.csv')
        self.test_annotations = pd.read_csv(
            f'{annotations_path}/EPIC_100_test_timestamps.csv')
        self.data_path = data_path
        
    def get_annotation_snippets(self, annotation, num_segments):
        participant_id = annotation[0]
        video_id = annotation[1]
        start_frame = annotation[2]
        stop_frame = annotation[3]
        zero = '0'
        transform = transforms.Compose(
            [transforms.PILToTensor(), transforms.Resize((224, 224))])
        segments = np.array_split(
            np.arange(start_frame, stop_frame + 1), num_segments)
        snippets = list(
            map(lambda x: str(np.random.default_rng().choice(x)), segments))
        file_names = list(
            map(lambda x: f'{(10 - len(x)) * zero}{x}', snippets))
        frames = torch.stack(list(map(lambda x: transform(Image.open(
            f'{self.dataset_path}/{participant_id}/rgb_frames/{video_id}/frame_{x}.jpg')), file_names)))
        return frames

    # TODO Change this to read in entire training set when I have it on hand.

    def get_train(self, num_segments):
        annotations = self.train_annotations.loc[0:328, ['participant_id',
                                                         'video_id', 'start_frame', 'stop_frame']].values
        train_X = torch.stack(
            list(map(lambda x: self.get_annotation_snippets(x, num_segments), annotations)))
        train_Y = torch.tensor(self.train_annotations.loc[0:328, [
            'verb_class', 'noun_class']].values)
        return train_X, train_Y

    def save_to_pt(self, filename, tensor):
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        torch.save(tensor, f'{self.data_path}/{filename}')

    def load_from_pt(self, filename):
        return torch.load(f'{self.data_path}/{filename}')
