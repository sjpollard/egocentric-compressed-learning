import argparse

import numpy as np
import pandas as pd
import tensorly as tl
import torch
import os
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

tl.set_backend('pytorch')

parser = argparse.ArgumentParser(
    description="Test the instantiation and forward pass of models",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--label",
    default="EPIC",
    type=str,
    help="Label prepended to the pytorch data files"
)
parser.add_argument(
    "--num-annotations",
    default=1000,
    type=int,
    help="Number of annotations to take from the csv file"
)
parser.add_argument(
    "--ratio",
    nargs=3,
    default=[80, 10, 10],
    type=int,
    help="Ratio of train/val/test splits respectively, input as space separated numbers that add to 100"
)
parser.add_argument(
    "--segment-count",
    default=8,
    type=int,
    help="Number of segments to pull clips from"
)
parser.add_argument(
    "--dataset-path",
    default="",
    type=str,
    help="Path to the EPIC-KITCHENS folder on the device"
)
parser.add_argument(
    "--seed",
    default=0,
    type=int,
    help="Random seed used to generate train/val/test splits"
)

class PreprocessedEPICDataset(Dataset):
    def __init__(self, dataset):
        assert dataset[0].size(0) == dataset[1].size(0)
        self.x = dataset[0]
        self.y = dataset[1]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.size(0)

class PostprocessedEPICDataset(Dataset):
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
        frames = []
        for i in range(self.num_segments):
            snippet = str(np.random.default_rng().choice(segments[i]))
            file_name = f'{(10 - len(snippet)) * zero}{snippet}'
            frame = self.transform(Image.open(f'{self.dataset_path}/{participant_id}/rgb_frames/{video_id}/frame_{file_name}.jpg'))
            frames.append(frame)
        return torch.stack(frames), self.annotations.loc[:, ['verb_class', 'noun_class']].values[index]

    def __len__(self):
        return len(self.annotations)

class DataProcessor:
    def __init__(self, dataset_path, annotations_path, data_path):
        self.dataset_path = dataset_path
        self.annotations = pd.read_csv(
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
        frames = []
        for i in range(self.num_segments):
            snippet = str(np.random.default_rng().choice(segments[i]))
            file_name = f'{(10 - len(snippet)) * zero}{snippet}'
            frame = transform(Image.open(f'{self.dataset_path}/{participant_id}/rgb_frames/{video_id}/frame_{file_name}.jpg'))
            frames.append(frame)
        return torch.stack(frames)

    def get_split(self, split, num_segments):
        annotations = split.loc[:, ['participant_id', 'video_id', 'start_frame', 'stop_frame']].values
        split_X = torch.stack(
            list(map(lambda x: self.get_annotation_snippets(x, num_segments), annotations)))
        split_Y = torch.tensor(split.loc[:, ['verb_class', 'noun_class']].values)
        return split_X, split_Y

    def split_annotations(self, num_annotations, ratio, seed):
        train_size = ratio[0]/100.0
        train, temp = train_test_split(self.annotations[:num_annotations], train_size=train_size , random_state=seed)
        val_size = (ratio[1]/100.0) / (ratio[1]/100.0 + ratio[2]/100.0)
        val, test = train_test_split(temp, train_size=val_size, random_state=seed)
        return train, val, test
    
    def save_to_pt(self, filename, tensor):
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        torch.save(tensor, f'{self.data_path}/{filename}')

    def load_from_pt(self, filename):
        return torch.load(f'{self.data_path}/{filename}')

def preprocess_epic(label, num_annotations, ratio, preprocessor, segment_count, seed):
    train, val, test = preprocessor.split_annotations(num_annotations, ratio, seed)
    train_X, train_Y = preprocessor.get_split(train.reset_index(), segment_count)
    preprocessor.save_to_pt(f'{label}_train_X.pt', train_X)
    preprocessor.save_to_pt(f'{label}_train_Y.pt', train_Y)
    val_X, val_Y = preprocessor.get_split(val.reset_index(), segment_count)
    preprocessor.save_to_pt(f'{label}_val_X.pt', val_X)
    preprocessor.save_to_pt(f'{label}_val_Y.pt', val_Y)
    test_X, test_Y = preprocessor.get_split(test.reset_index(), segment_count)
    preprocessor.save_to_pt(f'{label}_test_X.pt', test_X)
    preprocessor.save_to_pt(f'{label}_test_Y.pt', test_Y)

def main(args):
    preprocessor = DataProcessor(args.dataset_path, 'annotations', 'data')
    preprocess_epic(args.label, args.num_annotations, tuple(args.ratio), preprocessor, args.segment_count, args.seed)
   

if __name__ == "__main__":
    main(parser.parse_args())

