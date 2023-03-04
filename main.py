#!/usr/bin/env python3
import argparse
import logging
import sys

from pathlib import Path
from typing import Any, Dict
from numpy import matrix

import torch
import tensorly as tl
import data
import wandb
import pandas as pd
import compress
import os

from model_loader import load_checkpoint, make_model
from data import PreprocessedEPICDataset, PostprocessedEPICDataset
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.optim.optimizer import Optimizer
from torchvision import transforms
from ops.utils import compute_accuracy

tl.set_backend('pytorch')

torch.backends.cudnn.benchmark = True

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

parser = argparse.ArgumentParser(
    description="Test the instantiation and forward pass of models",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "model_type",
    nargs="?",
    choices=["tsn", "tsm", "tsm-nl", "trn", "mtrn"],
    default=None,
)
parser.add_argument(
    "--checkpoint",
    type=Path,
    help="Path to checkpointed model. Should be a dictionary containing the keys:"
    " 'model_type', 'segment_count', 'modality', 'state_dict', and 'arch'.",
)
parser.add_argument(
    "--arch",
    default="resnet50",
    choices=["BNInception", "resnet50"],
    help="Backbone architecture",
)
parser.add_argument(
    "--modality", default="RGB", choices=["RGB", "Flow"], help="Input modality"
)
parser.add_argument(
    "--flow-length", default=5, type=int, help="Number of (u, v) pairs in flow stack"
)
parser.add_argument(
    "--dropout",
    default=0.7,
    type=float,
    help="Dropout probability. The dropout layer replaces the "
    "backbone's classification layer.",
)
parser.add_argument(
    "--trn-img-feature-dim",
    default=256,
    type=int,
    help="Number of dimensions for the output of backbone network. "
    "This is effectively the image feature dimensionality.",
)
parser.add_argument(
    "--segment-count",
    default=8,
    type=int,
    help="Number of segments. For RGB this corresponds to number of "
    "frames, whereas for Flow, it is the number of points from "
    "which a stack of (u, v) frames are sampled.",
)
parser.add_argument(
    "--tsn-consensus-type",
    choices=["avg", "max"],
    default="avg",
    help="Consensus function for TSN used to fuse class scores from "
    "each segment's predictoin.",
)
parser.add_argument(
    "--tsm-shift-div",
    default=8,
    type=int,
    help="Reciprocal proportion of features temporally-shifted.",
)
parser.add_argument(
    "--tsm-shift-place",
    default="blockres",
    choices=["block", "blockres"],
    help="Location for the temporal shift to take place. Either 'block' for the shift "
    "to happen in the non-residual part of a block, or 'blockres' if the shift happens "
    "in the residual path.",
)
parser.add_argument(
    "--tsm-temporal-pool",
    action="store_true",
    help="Gradually temporally pool throughout the network",
)
parser.add_argument(
    "--load",
    default="preprocessed",
    choices=["preprocessed", "postprocessed"],
    help="Use 'preprocessed' or 'postprocessed' dataset",
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
    "--num-annotations",
    default=1000,
    type=int,
    help="Number of annotations to postprocess from EPIC-KITCHENS"
)
parser.add_argument(
    "--chunks",
    default=1,
    type=int,
    help="Number of evenly sized chunks in preprocessed dataset"
)
parser.add_argument(
    "--ratio",
    nargs=3,
    default=[80, 10, 10],
    type=int,
    help="Ratio of train/val/test splits respectively in postprocessed dataset"
)
parser.add_argument(
    "--seed",
    default=0,
    type=int,
    help="Random seed used to generate train/val/test splits"
)
parser.add_argument(
    "--epochs",
    default=10,
    type=int,
    help="Number of epochs to train"
)
parser.add_argument(
    "--batch-size",
    default=10,
    type=int,
    help="Number of clips per batch"
)
parser.add_argument(
    "--lr", 
    default=1e-3, 
    type=float, 
    help="Learning rate of the network"
)
parser.add_argument(
    "--val-frequency", 
    default=1, 
    type=int, 
    help="Epochs until validation set is tested"
)
parser.add_argument(
    "--log-frequency", 
    default=0, 
    type=int, 
    help="Steps until logs are saved with `wandb`"
)
parser.add_argument(
    "--print-frequency", 
    default=10, 
    type=int, 
    help="Steps until training batch results are printed"
)
parser.add_argument(
    "--print-model", 
    action="store_true", 
    help="Print model definition"
)
parser.add_argument(
    "--save-model",
    default=False,
    action="store_true",
    help="Saves model as checkpoint for evaluation"
)


def extract_settings_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    settings = vars(args)
    for variant in ["trn", "tsm", "tsn"]:
        variant_key_prefix = f"{variant}_"
        variant_keys = {
            key for key in settings.keys() if key.startswith(variant_key_prefix)
        }
        for key in variant_keys:
            stripped_key = key[len(variant_key_prefix) :]
            settings[stripped_key] = settings[key]
            del settings[key]
    return settings


def get_dataloaders(dataprocessor, args):
    if args.load == 'preprocessed':
        train_dataset = PreprocessedEPICDataset(dataprocessor, args.label, args.chunks, 'train')
        val_dataset = PreprocessedEPICDataset(dataprocessor, args.label, args.chunks, 'val')
        test_dataset = PreprocessedEPICDataset(dataprocessor, args.label, args.chunks, 'test')
    elif args.load == 'postprocessed':
        train, val, test = dataprocessor.split_annotations(args.num_annotations, tuple(args.ratio), args.seed)
        train_dataset = PostprocessedEPICDataset(args.dataset_path, train.reset_index(), 
                                 transforms.Compose([transforms.PILToTensor(), transforms.Resize((224, 224))]),
                                 args.segment_count)
        val_dataset = PostprocessedEPICDataset(args.dataset_path, val.reset_index(), 
                                 transforms.Compose([transforms.PILToTensor(), transforms.Resize((224, 224))]),
                                 args.segment_count)
        test_dataset = PostprocessedEPICDataset(args.dataset_path, test.reset_index(), 
                                 transforms.Compose([transforms.PILToTensor(), transforms.Resize((224, 224))]),
                                 args.segment_count)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader, test_dataloader

class Trainer:
    def __init__(self, 
                 model: nn.Module, 
                 train_dataloader: DataLoader, 
                 val_dataloader: DataLoader,
                 test_dataloader: DataLoader,
                 criterion: nn.Module, 
                 optimizer: Optimizer,
                 phi_matrices: list,
                 modes: list):
        self.model = model.to(DEVICE)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.phi_matrices = phi_matrices
        self.modes = modes
    
    def train(self, epochs, val_frequency, log_frequency, print_frequency):
        self.model.train()
        for epoch in range(1, epochs + 1):
            self.step = 1
            self.model.train()
            for x, y in self.train_dataloader:
                x = x.float().to(DEVICE)
                y = y.to(DEVICE)
                if self.phi_matrices != None: compress.process_batch(x, self.phi_matrices, self.modes)
                verb_output, noun_output = self.model(x)
                verb_loss = self.criterion(verb_output, y[:, 0])
                noun_loss = self.criterion(noun_output, y[:, 1])
                loss = verb_loss + noun_loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                with torch.no_grad():
                    y_hat_verb = torch.argmax(verb_output, dim=-1)
                    y_hat_noun = torch.argmax(noun_output, dim=-1)
                    verb_accuracy = compute_accuracy(y[:, 0], y_hat_verb)
                    noun_accuracy = compute_accuracy(y[:, 1], y_hat_noun)
                if log_frequency != 0 and (self.step % log_frequency) == 0:
                    wandb.log({'train/verb-loss': verb_loss,
                               'train/noun-loss': noun_loss, 
                               'train/verb-accuracy': verb_accuracy, 
                               'train/noun-accuracy': noun_accuracy})
                if (self.step % print_frequency) == 0:
                    self.print_metrics(epoch, verb_loss, noun_loss, verb_accuracy, noun_accuracy)
                self.step += 1
            if (epoch % val_frequency) == 0:
                self.validate('val', log_frequency)
                self.model.train()
    
    def validate(self, split, log_frequency):
        if split == 'val': split_dataloader = self.val_dataloader
        elif split == 'test': split_dataloader = self.test_dataloader
        self.model.eval()
        total_verb_loss = 0
        total_noun_loss = 0
        ys = []
        y_hats = []
        with torch.no_grad():
            for x, y in split_dataloader:
                x = x.float().to(DEVICE)
                y = y.to(DEVICE)
                if self.phi_matrices != None: compress.process_batch(x, self.phi_matrices, self.modes)
                verb_output, noun_output = self.model(x)
                verb_loss = self.criterion(verb_output, y[:, 0])
                noun_loss = self.criterion(noun_output, y[:, 1])
                total_verb_loss += verb_loss.item()
                total_noun_loss += noun_loss.item()
                y_hat_verb = torch.argmax(verb_output, dim=-1)
                y_hat_noun = torch.argmax(noun_output, dim=-1)
                ys.append(y)
                y_hats.append(torch.stack((y_hat_verb, y_hat_noun), 1))
                
        ys = torch.cat(ys)
        y_hats = torch.cat(y_hats)
        verb_accuracy = compute_accuracy(ys[:, 0], y_hats[:, 0])
        noun_accuracy = compute_accuracy(ys[:, 1], y_hats[:, 1])

        average_verb_loss = total_verb_loss / len(split_dataloader)
        average_noun_loss = total_noun_loss / len(split_dataloader)

        if (log_frequency != 0):
            wandb.log({f'{split}/avg-verb-loss': average_verb_loss,
                    f'{split}/avg-noun-loss': average_noun_loss,
                    f'{split}/verb-accuracy': verb_accuracy, 
                    f'{split}/noun-accuracy': noun_accuracy})
        print(f"{split}: avg verb loss: {average_verb_loss:.5f} avg noun loss: {average_noun_loss:.5f}, verb accuracy: {verb_accuracy * 100:2.2f} noun accuracy: {noun_accuracy * 100:2.2f}")

    def print_metrics(self, epoch, verb_loss, noun_loss, verb_accuracy, noun_accuracy):
        epoch_step = self.step % len(self.train_dataloader)
        if epoch_step == 0: epoch_step = len(self.train_dataloader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_dataloader)}], "
                f"batch verb loss: {verb_loss:.5f}",
                f"batch noun loss: {noun_loss:.5f}, "
                f"batch verb accuracy: {verb_accuracy * 100:2.2f}",
                f"batch noun accuracy: {noun_accuracy * 100:2.2f} "
        )

def main(args):
    dataprocessor = data.DataProcessor(args.dataset_path, 'annotations', 'data')

    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(dataprocessor, args)
    
    logging.basicConfig(level=logging.INFO)
    if args.checkpoint is None:
        if args.model_type is None:
            print("If not providing a checkpoint, you must specify model_type")
            sys.exit(1)
        settings = extract_settings_from_args(args)
        model = make_model(settings)
    elif args.checkpoint is not None and args.checkpoint.exists():
        model = load_checkpoint(args.checkpoint)
    else:
        print(f"{args.checkpoint} doesn't exist")
        sys.exit(1)

    if args.print_model:
        print(model)
    
    if args.log_frequency != 0:
        wandb.init(project="egocentric-compressed-learning", config=settings)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    clip_dims = train_dataloader.dataset.__getitem__(0)[0].size()
    if args.matrix_type == 'bernoulli':
        matrix_gen = compress.random_bernoulli_matrix
    elif args.matrix_type == 'gaussian':
        matrix_gen = compress.random_gaussian_matrix
    phi_matrices = None if args.matrix_type == None else list(map(lambda x, y: 
                        matrix_gen((x, clip_dims[y])).to(DEVICE), args.measurements, args.modes))

    trainer = Trainer(model, train_dataloader, val_dataloader, test_dataloader, criterion, optimizer, phi_matrices, args.modes)

    trainer.train(args.epochs, args.val_frequency, args.log_frequency, args.print_frequency)
    trainer.validate('test', args.log_frequency)

    if args.save_model != None:
        if args.matrix_type == None:
            filename = f'{args.label}_{args.epochs}.pt'
        else:
            filename = f'{args.label}_{args.matrix_type}_{"_".join(map(str, args.measurements))}_{"_".join(map(str, args.modes))}_{args.epochs}.pt'
            phi_matrices_filename = f'phi_matrices_{args.label}_{args.matrix_type}_{"_".join(map(str, args.measurements))}_{"_".join(map(str, args.modes))}_{args.epochs}.pt'
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        torch.save(trainer.model.state_dict(), f'checkpoints/{filename}')
        if args.matrix_type != None: torch.save(phi_matrices, f'checkpoints/{phi_matrices_filename}')

if __name__ == "__main__":
    main(parser.parse_args())
