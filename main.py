#!/usr/bin/env python3
import argparse
import logging
import sys

from pathlib import Path
from typing import Any, Dict

import torch
import tensorly as tl
import data
import wandb

from model_loader import load_checkpoint, make_model
from data import PreprocessedEPICDataset
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.optim.optimizer import Optimizer


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
parser.add_argument("--batch-size", default=10, type=int, help="Batch size")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs to train")
parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
parser.add_argument("--val-frequency", default=2, type=int, help="How frequently to test the model on the validation set in number of epochs")
parser.add_argument("--log-frequency", default=10, type=int, help="How frequently to save logs to wandb in number of steps",)
parser.add_argument("--print-frequency", default=10, type=int, help="How frequently to print progress to the command line in number of steps")
parser.add_argument("--print-model", action="store_true", help="Print model definition")


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


def preprocess_epic(label, num_annotations, ratio, preprocessor, segment_count):
    train, val, test = preprocessor.split_annotations(num_annotations, ratio, 0)
    train_X, train_Y = preprocessor.get_split(train, segment_count)
    preprocessor.save_to_pt(f'{label}_train_X.pt', train_X)
    preprocessor.save_to_pt(f'{label}_train_Y.pt', train_Y)
    val_X, val_Y = preprocessor.get_split(val, segment_count)
    preprocessor.save_to_pt(f'{label}_val_X.pt', val_X)
    preprocessor.save_to_pt(f'{label}_val_Y.pt', val_Y)
    test_X, test_Y = preprocessor.get_split(test, segment_count)
    preprocessor.save_to_pt(f'{label}_test_X.pt', test_X)
    preprocessor.save_to_pt(f'{label}_test_Y.pt', test_Y)


def compute_accuracy(y, y_hat):
    assert len(y) == len(y_hat)
    return float((y == y_hat).sum()) / len(y)


class Trainer:
    def __init__(self, 
                 model: nn.Module, 
                 train_dataloader: DataLoader, 
                 val_dataloader: DataLoader, 
                 criterion: nn.Module, 
                 optimizer: Optimizer):
        self.model = model.to(DEVICE)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.step = 0
    
    def train(self, epochs, val_frequency, log_frequency, print_frequency):
        self.model.train()
        for epoch in range(epochs):
            self.model.train()
            for x, y in self.train_dataloader:
                x = x.float().to(DEVICE)
                y = y.to(DEVICE)
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
                if ((self.step + 1) % log_frequency) == 0:
                    wandb.log({'train/verb-loss': verb_loss,
                               'train/noun-loss': noun_loss, 
                               'train/verb-accuracy': verb_accuracy, 
                               'train/noun-accuracy': noun_accuracy})
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, verb_loss, noun_loss, verb_accuracy, noun_accuracy)
                self.step += 1
            if ((epoch + 1) % val_frequency) == 0:
                self.validate()
                self.model.train()
    
    def validate(self):
        self.model.eval()
        total_verb_loss = 0
        total_noun_loss = 0
        ys_verb = torch.empty(self.val_dataloader.batch_size).to(DEVICE)
        ys_noun = torch.empty(self.val_dataloader.batch_size).to(DEVICE)
        y_hats_verb = torch.empty(self.val_dataloader.batch_size).to(DEVICE)
        y_hats_noun = torch.empty(self.val_dataloader.batch_size).to(DEVICE)
        with torch.no_grad():
            for x, y in self.val_dataloader:
                x = x.float().to(DEVICE)
                y = y.to(DEVICE)
                verb_output, noun_output = self.model(x)
                verb_loss = self.criterion(verb_output, y[:, 0])
                noun_loss = self.criterion(noun_output, y[:, 1])
                total_verb_loss += verb_loss.item()
                total_noun_loss += noun_loss.item()
                y_hat_verb = torch.argmax(verb_output, dim=-1)
                y_hat_noun = torch.argmax(noun_output, dim=-1)
                ys_verb = torch.cat((ys_verb, y[:, 0]))
                ys_noun = torch.cat((ys_noun, y[:, 1]))
                y_hats_verb = torch.cat((y_hats_verb, y_hat_verb))
                y_hats_noun = torch.cat((y_hats_noun, y_hat_noun))

        verb_accuracy = compute_accuracy(ys_verb, y_hats_verb)
        noun_accuracy = compute_accuracy(ys_noun, y_hats_noun)

        average_verb_loss = total_verb_loss / len(self.val_dataloader)
        average_noun_loss = total_noun_loss / len(self.val_dataloader)

        wandb.log({'val/avg-verb-loss': average_verb_loss,
                   'val/avg-noun-loss': average_noun_loss,
                   'val/verb-accuracy': verb_accuracy, 
                   'val/noun-accuracy': noun_accuracy})
        print(f"validation: avg verb loss: {average_verb_loss:.5f}, avg noun loss: {average_noun_loss:.5f}, verb accuracy: {verb_accuracy * 100:2.2f}, noun accuracy: {noun_accuracy * 100:2.2f}")

    def print_metrics(self, epoch, verb_loss, noun_loss, verb_accuracy, noun_accuracy):
        epoch_step = self.step % len(self.train_dataloader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_dataloader)}], "
                f"batch verb loss: {verb_loss:.5f}, ",
                f"batch noun loss: {noun_loss:.5f}, "
                f"batch verb accuracy: {verb_accuracy * 100:2.2f}",
                f"batch noun accuracy: {noun_accuracy * 100:2.2f}, "
        )

def main(args):
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
    height, width = model.input_size, model.input_size
    if model.modality == "RGB":
        channel_dim = 3
    elif model.modality == "Flow":
        channel_dim = args.flow_length * 2
    else:
        raise ValueError(f"Unknown modality {args.modality}")
    
    wandb.init(project="egocentric-compressed-learning", config=settings)

    """ dataset = PostprocessedEPICDataset('C:/Users/SAM/EPIC-KITCHENS', pd.read_csv('annotations/EPIC.csv'), 
                                 transforms.Compose([transforms.PILToTensor(), transforms.Resize((224, 224))]),
                                 8)
    train_dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True) """

    preprocessor = data.Preprocessor('B:/EPIC-KITCHENS', 'annotations', 'data')
    
    #preprocess_epic('P01_P02', 6215, (80, 10, 10), preprocessor, 8)

    train_X, train_Y = preprocessor.load_from_pt('P01_P02_train_X.pt'), preprocessor.load_from_pt('P01_P02_train_Y.pt')
    val_X, val_Y = preprocessor.load_from_pt('P01_P02_val_X.pt'), preprocessor.load_from_pt('P01_P02_val_Y.pt')
    train_dataset = PreprocessedEPICDataset(dataset=(train_X, train_Y))
    val_dataset = PreprocessedEPICDataset(dataset=(val_X, val_Y))
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    trainer = Trainer(model, train_dataloader, val_dataloader, criterion, optimizer)

    trainer.train(args.epochs, args.val_frequency, args.log_frequency, args.print_frequency)

    """ torchvision.transforms.functional.to_pil_image(input[0][3]).show()
    M1 = compress.random_bernoulli_matrix((100, 224))
    M2 = compress.random_bernoulli_matrix((100, 224))
    compressed_clip = compress.compress_tensor(input[0], [M1, M2], [3, 2])
    torchvision.transforms.functional.to_pil_image(compressed_clip[3]).show()
    input[0] = compress.expand_tensor(compressed_clip, [M1.T, M2.T], [3, 2])
    torchvision.transforms.functional.to_pil_image(clips[0][3]).show() """

if __name__ == "__main__":
    main(parser.parse_args())
