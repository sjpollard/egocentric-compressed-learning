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
from ops.utils import compute_accuracy

tl.set_backend('pytorch')

parser = argparse.ArgumentParser(
    description="Handles processing and chunking of dataset",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--dataset-path",
    default="",
    type=str,
    help="Path to the EPIC-KITCHENS folder on the device"
)
parser.add_argument(
    "--label",
    default="EPIC",
    type=str,
    help="Label to prepend to preprocessed dataset files"
)
parser.add_argument(
    "--num-annotations",
    default=1000,
    type=int,
    help="Number of annotations to preprocess from EPIC-KITCHENS"
)
parser.add_argument(
    "--chunks",
    default=1,
    type=int,
    help="Number of evenly sized chunks to preprocess into"
)
parser.add_argument(
    "--ratio",
    nargs=3,
    default=[80, 10, 10],
    type=int,
    help="Ratio of train/val/test splits respectively in preprocessed dataset"
)
parser.add_argument(
    "--seed",
    default=0,
    type=int,
    help="Random seed used to generate train/val/test splits"
)
parser.add_argument(
    "--segment-count",
    default=8,
    type=int,
    help="Number of temporal segments to sample from"
)
parser.add_argument(
    "--random-perf",
    default=False,
    action="store_true",
    help="Print random performance of dataset splits"
)

class PreprocessedEPICDataset(Dataset):
    def __init__(self, dataprocessor, label, chunks, split_type):
        self.dataprocessor = dataprocessor
        self.label = label
        self.chunks = chunks
        self.split_type = split_type
        self.length = 0
        for i in range(1, chunks + 1):
            self.length += dataprocessor.load_from_pt(label, f'{label}_{i}_{split_type}_Y.pt').size(0)
        self.chunk_reference = {}
        chunk_index = 1
        for chunk in np.array_split(np.arange(self.length), chunks):
            element_index = 0
            for element in chunk:
                self.chunk_reference.update({str(element): {'chunk_index': chunk_index,'element_index': element_index}})
                element_index += 1
            chunk_index += 1

    def __getitem__(self, index):
        chunk_index = self.chunk_reference[str(index)]['chunk_index']
        element_index = self.chunk_reference[str(index)]['element_index']
        x = self.dataprocessor.load_from_pt(self.label, f'{self.label}_{chunk_index}_{self.split_type}_X.pt')[element_index]
        y = self.dataprocessor.load_from_pt(self.label, f'{self.label}_{chunk_index}_{self.split_type}_Y.pt')[element_index]
        return x, y

    def __len__(self):
        return self.length

class PostprocessedEPICDataset(Dataset):
    def __init__(self, dataset_path, annotations, transform, segment_count):
        self.dataset_path = dataset_path
        self.annotations = annotations
        self.transform = transform
        self.segment_count = segment_count

    def __getitem__(self, index):
        participant_id = self.annotations.at[index, 'participant_id']
        video_id = self.annotations.at[index, 'video_id']
        start_frame = self.annotations.at[index, 'start_frame']
        stop_frame = self.annotations.at[index, 'stop_frame']
        zero = '0'
        segments = np.array_split(
                np.arange(start_frame, stop_frame + 1), self.segment_count)
        frames = []
        for i in range(self.segment_count):
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
        
    def get_annotation_snippets(self, annotation, segment_count):
        participant_id = annotation[0]
        video_id = annotation[1]
        start_frame = annotation[2]
        stop_frame = annotation[3]
        zero = '0'
        transform = transforms.Compose(
            [transforms.PILToTensor(), transforms.Resize((224, 224))])
        segments = np.array_split(
            np.arange(start_frame, stop_frame + 1), segment_count)
        frames = []
        for i in range(segment_count):
            snippet = str(np.random.default_rng().choice(segments[i]))
            file_name = f'{(10 - len(snippet)) * zero}{snippet}'
            frame = transform(Image.open(f'{self.dataset_path}/{participant_id}/rgb_frames/{video_id}/frame_{file_name}.jpg'))
            frames.append(frame)
        return torch.stack(frames)

    def get_split(self, split, segment_count):
        annotations = split.loc[:, ['participant_id', 'video_id', 'start_frame', 'stop_frame']].values
        split_X = torch.stack(
            list(map(lambda x: self.get_annotation_snippets(x, segment_count), annotations)))
        split_Y = torch.tensor(split.loc[:, ['verb_class', 'noun_class']].values)
        return split_X, split_Y

    def split_annotations(self, num_annotations, ratio, seed):
        train_size = ratio[0]/100.0
        train, temp = train_test_split(self.annotations[:num_annotations], train_size=train_size , random_state=seed)
        val_size = (ratio[1]/100.0) / (ratio[1]/100.0 + ratio[2]/100.0)
        val, test = train_test_split(temp, train_size=val_size, random_state=seed)
        return train, val, test
    
    def save_to_pt(self, label, filename, tensor):
        if not os.path.exists(f'{self.data_path}/{label}'):
            os.makedirs(f'{self.data_path}/{label}')
        torch.save(tensor, f'{self.data_path}/{label}/{filename}')

    def load_from_pt(self, label, filename):
        return torch.load(f'{self.data_path}/{label}/{filename}')
    
    def load_annotations(self, label):
        train = pd.read_csv(f'{self.data_path}/{label}/{label}_train.csv')
        val = pd.read_csv(f'{self.data_path}/{label}/{label}_val.csv')
        test = pd.read_csv(f'{self.data_path}/{label}/{label}_test.csv')
        return train, val, test

def preprocess_epic(label, num_annotations, chunks, ratio, dataprocessor, segment_count, seed):
    train, val, test = dataprocessor.split_annotations(num_annotations, ratio, seed)
    train_chunks = np.array_split(train, chunks)
    for i in range(len(train_chunks)):
        chunk_X, chunk_Y = dataprocessor.get_split(train_chunks[i].reset_index(), segment_count)
        dataprocessor.save_to_pt(label, f'{label}_{i + 1}_train_X.pt', chunk_X)
        dataprocessor.save_to_pt(label, f'{label}_{i + 1}_train_Y.pt', chunk_Y)
    val_chunks = np.array_split(val, chunks)
    for i in range(len(val_chunks)):
        chunk_X, chunk_Y = dataprocessor.get_split(val_chunks[i].reset_index(), segment_count)
        dataprocessor.save_to_pt(label, f'{label}_{i + 1}_val_X.pt', chunk_X)
        dataprocessor.save_to_pt(label, f'{label}_{i + 1}_val_Y.pt', chunk_Y)
    test_chunks = np.array_split(test, chunks)
    for i in range(len(test_chunks)):
        chunk_X, chunk_Y = dataprocessor.get_split(test_chunks[i].reset_index(), segment_count)
        dataprocessor.save_to_pt(label, f'{label}_{i + 1}_test_X.pt', chunk_X)
        dataprocessor.save_to_pt(label, f'{label}_{i + 1}_test_Y.pt', chunk_Y)
    train.reset_index(drop=True).to_csv(f'{dataprocessor.data_path}/{label}/{label}_train.csv', index=False)
    val.reset_index(drop=True).to_csv(f'{dataprocessor.data_path}/{label}/{label}_val.csv', index=False)
    test.reset_index(drop=True).to_csv(f'{dataprocessor.data_path}/{label}/{label}_test.csv', index=False)

def main(args):
    dataprocessor = DataProcessor(args.dataset_path, 'annotations', 'data')

    if args.random_perf:
        train, val, test = dataprocessor.load_annotations(args.label)

        train_counts = train['verb_class'].value_counts(normalize=True), train['noun_class'].value_counts(normalize=True)
        train_classes = train_counts[0].index.to_numpy(), train_counts[1].index.to_numpy()
        train_probs = train_counts[0].to_numpy(), train_counts[1].to_numpy()
        train_ys = train[['verb_class', 'noun_class']].to_numpy()
        train_y_hats = np.random.default_rng().choice(train_classes[0], train_ys.shape[0], p=train_probs[0]), np.random.default_rng().choice(train_classes[1], train_ys.shape[0], p=train_probs[1])
        print(f'train verb {compute_accuracy(train_ys[:, 0], train_y_hats[0])}, train noun {compute_accuracy(train_ys[:, 1], train_y_hats[1])}')
        
        val_counts = val['verb_class'].value_counts(normalize=True), val['noun_class'].value_counts(normalize=True)
        val_classes = val_counts[0].index.to_numpy(), val_counts[1].index.to_numpy()
        val_probs = val_counts[0].to_numpy(), val_counts[1].to_numpy()
        val_ys = val[['verb_class', 'noun_class']].to_numpy()
        val_y_hats = np.random.default_rng().choice(val_classes[0], val_ys.shape[0], p=val_probs[0]), np.random.default_rng().choice(val_classes[1], val_ys.shape[0], p=val_probs[1])
        print(f'val verb {compute_accuracy(val_ys[:, 0], val_y_hats[0])}, val noun {compute_accuracy(val_ys[:, 1], val_y_hats[1])}')

        test_counts = test['verb_class'].value_counts(normalize=True), test['noun_class'].value_counts(normalize=True)
        test_classes = test_counts[0].index.to_numpy(), test_counts[1].index.to_numpy()
        test_probs = test_counts[0].to_numpy(), test_counts[1].to_numpy()
        test_ys = test[['verb_class', 'noun_class']].to_numpy()
        test_y_hats = np.random.default_rng().choice(test_classes[0], test_ys.shape[0], p=test_probs[0]), np.random.default_rng().choice(test_classes[1], test_ys.shape[0], p=test_probs[1])
        print(f'test verb {compute_accuracy(test_ys[:, 0], test_y_hats[0])}, test noun {compute_accuracy(test_ys[:, 1], test_y_hats[1])}')
    else:
        preprocess_epic(args.label, args.num_annotations, args.chunks, tuple(args.ratio), dataprocessor, args.segment_count, args.seed)
   

if __name__ == "__main__":
    main(parser.parse_args())
