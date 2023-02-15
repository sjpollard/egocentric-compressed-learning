import numpy as np
import pandas as pd
import tensorly as tl
import torch
import os
import tarfile
import io
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

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

class EPICDataset(Dataset):
  def __init__(self, dataset_path, annotations, transform, num_segments):
    self.dataset_path = dataset_path
    self.annotations = annotations[:329]
    self.transform = transform
    self.num_segments = num_segments

  def __getitem__(self, index):
    participant_id = self.annotations.at[index, 'participant_id']
    video_id = self.annotations.at[index, 'video_id']
    start_frame = self.annotations.at[index, 'start_frame']
    stop_frame = self.annotations.at[index, 'stop_frame']
    zero = '0'
    segments = np.array_split(
            np.arange(start_frame, stop_frame + 1), self.num_segments)
    snippets = list(
        map(lambda x: str(np.random.default_rng().choice(x)), segments))
    file_names = list(
        map(lambda x: f'{(10 - len(x)) * zero}{x}', snippets))
    frames = torch.stack(list(map(lambda x: self.transform(Image.open(
            f'{self.dataset_path}/{participant_id}/rgb_frames/{video_id}/frame_{x}.jpg')), file_names)))
    return frames.float(), self.annotations.loc[:, ['verb_class', 'noun_class']].values[index]

  def __len__(self):
    return len(self.annotations)

class EPICTarDataset(Dataset):
  def __init__(self, dataset_path, annotations, transform, num_segments):
    self.dataset_path = dataset_path
    self.annotations = annotations
    self.transform = transform
    self.num_segments = num_segments

  def __getitem__(self, index):
    participant_id = self.annotations.at[index, 'participant_id']
    video_id = self.annotations.at[index, 'video_id']
    start_frame = self.annotations.at[index, 'start_frame']
    stop_frame = self.annotations.at[index, 'stop_frame']
    zero = '0'
    segments = np.array_split(
            np.arange(start_frame, stop_frame + 1), self.num_segments)
    snippets = list(
        map(lambda x: str(np.random.default_rng().choice(x)), segments))
    file_names = list(
        map(lambda x: f'{(10 - len(x)) * zero}{x}', snippets))
    with tarfile.open(f'{self.dataset_path}/{participant_id}/rgb_frames/{video_id}.tar') as tf:
        frames = torch.stack(list(map(lambda x: self.transform(Image.open(tf.extractfile(tf.getmember(f'./frame_{x}.jpg')))), file_names)))
    return frames, self.annotations.loc[:, ['verb_class', 'noun_class']].values

  def __len__(self):
    return len(self.annotations)

class Preprocessor:
    def __init__(self, dataset_path, annotations_path, data_path):
        self.dataset_path = dataset_path
        self.epic_annotations = pd.read_csv(
            f'{annotations_path}/EPIC.csv')
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

    def get_split(self, split, num_segments):
        annotations = self.split.loc[:, ['participant_id', 'video_id', 'start_frame', 'stop_frame']].values
        split_X = torch.stack(
            list(map(lambda x: self.get_annotation_snippets(x, num_segments), annotations)))
        split_Y = torch.tensor(self.split.loc[:, ['verb_class', 'noun_class']].values)
        return split_X, split_Y

    def split_annotations(self, ratio, seed):
        train_size = ratio[0]/100.0
        train, temp = train_test_split(self.epic_annotations, train_size=train_size , random_state=seed)
        val_size = (ratio[1]/100.0) / (ratio[1]/100.0 + ratio[2]/100.0)
        print(train_size, val_size)
        val, test = train_test_split(temp, train_size=val_size, random_state=seed)
        return train, val, test
    
    def save_to_pt(self, filename, tensor):
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        torch.save(tensor, f'{self.data_path}/{filename}')

    def load_from_pt(self, filename):
        return torch.load(f'{self.data_path}/{filename}')
