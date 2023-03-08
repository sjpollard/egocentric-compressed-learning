import argparse

import torch
import torchshow as ts
import data
import tensorly as tl

from data import PreprocessedEPICDataset

tl.set_backend('pytorch')

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

parser = argparse.ArgumentParser(
    description="Test the instantiation and forward pass of models",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--label",
    default="EPIC",
    type=str,
    help="Label prepended to preprocessed dataset files"
)
parser.add_argument(
    "--matrix-type",
    default=None,
    choices=[None, "bernoulli", "gaussian"],
    help="'bernoulli' or 'gaussian' matrices",
)
parser.add_argument(
    "--measurements", 
    nargs='*',
    default=None, 
    type=int, 
    help="Heights of measurement matrices"
)
parser.add_argument(
    "--modes", 
    nargs='*',
    default=None, 
    type=int, 
    help="Modes corresponding to measurement matrices"
)
parser.add_argument(
    "--chunks",
    default=1,
    type=int,
    help="Number of evenly sized chunks in preprocessed dataset"
)
parser.add_argument(
    "--epochs",
    default=10,
    type=int,
    help="Number of epochs to train"
)
parser.add_argument(
    "--split",
    choices=["train", "val", "test"],
    default="train",
    help="Dataset split to visualise from",
)
parser.add_argument(
    "--index",
    default=0,
    type=int,
    help="Index of clip to visualise"
)

def get_dataset(dataprocessor, args):
    dataset = PreprocessedEPICDataset(dataprocessor, args.label, args.chunks, args.split)
    return dataset

def main(args):
    dataprocessor = data.DataProcessor('', 'annotations', 'data')
    dataset = get_dataset(dataprocessor, args)
    clip = dataset.__getitem__(args.index)[0].float().to(DEVICE)
    ts.show(clip)
    if args.matrix_type != None:
        phi_matrices_filename = f'phi_matrices_{args.label}_{args.matrix_type}_{"_".join(map(str, args.measurements))}_{"_".join(map(str, args.modes))}_{args.epochs}.pt'
        phi_matrices = torch.load(f'checkpoints/{phi_matrices_filename}')
        ts.show(phi_matrices)
        compressed_clip = tl.tenalg.multi_mode_dot(clip, phi_matrices, args.modes)
        ts.show(compressed_clip)
        inferred_clip = tl.tenalg.multi_mode_dot(compressed_clip, phi_matrices, args.modes, transpose=True)
        ts.show(inferred_clip)

if __name__ == "__main__":
    main(parser.parse_args())
