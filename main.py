#!/usr/bin/env python3
import argparse
import logging
from pyexpat import model
import sys

from pathlib import Path
from typing import Any, Dict

import torch
import tensorly as tl
import torchvision
import compress
import data
import pandas as pd

from model_loader import load_checkpoint, make_model
from data import CustomClipDataset, EPICTarDataset, EPICDataset
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms


tl.set_backend('pytorch')

#torch.backends.cudnn.benchmark = True

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cuda')

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
parser.add_argument("--lr", default=1e-2, type=float, help="Learning rate")
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


def preprocess_epic(ratio, preprocessor, segment_count):
    train, val, test = preprocessor.split_annotations(ratio, 0)
    train_X, train_Y = preprocessor.get_split(train, segment_count)
    val_X, val_Y = preprocessor.get_split(val, segment_count)
    test_X, test_Y = preprocessor.get_split(test, segment_count)
    preprocessor.save_to_pt('train_X.pt', train_X)
    preprocessor.save_to_pt('train_Y.pt', train_Y)
    preprocessor.save_to_pt('val_X.pt', val_X)
    preprocessor.save_to_pt('val_Y.pt', val_Y)
    preprocessor.save_to_pt('test_X.pt', test_X)
    preprocessor.save_to_pt('test_Y.pt', test_Y)


def compute_accuracy(y, y_hat):
    assert len(y) == len(y_hat)
    return float((y == y_hat).sum()) / len(y)


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
    
    dataset = EPICDataset('C:/Users/SAM/EPIC-KITCHENS', pd.read_csv('annotations/EPIC.csv'), 
                                 transforms.Compose([transforms.PILToTensor(), transforms.Resize((224, 224))]),
                                 8)
    train_dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

    

    """ preprocessor = data.Preprocessor('C:/Users/SAM/EPIC-KITCHENS',
                                 'annotations',
                                 'data')
    preprocessor = data.Preprocessor('/home/hiraeth/EPIC-KITCHENS',
                               '/home/hiraeth/Github/epic-kitchens-100-annotations',
                               '/home/hiraeth/Github/egocentric-compressed-learning/data') """
    #preprocess_epic(data_loader)
    """ train_X, train_Y = preprocessor.load_from_pt('train_X.pt').float(), preprocessor.load_from_pt('train_Y.pt')
    train_dataset = CustomClipDataset(dataset=(train_X, train_Y))
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True) """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    model.to(DEVICE)
    model.train()
    for epoch in range(args.epochs):
        print("Epoch: ", epoch)
        model.train()
        i = 0
        for x, y in train_dataloader:
            if i % 10 == 0 : print("Batch: ", i)
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            verb_output,_ = model(x)
            verb_loss = criterion(verb_output, y[:, 0])
            verb_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            i += 1
            with torch.no_grad():
                print("Batch accuracy: ", compute_accuracy(y[:, 0], torch.argmax(verb_output, dim=-1)))

    #print("Total accuracy: ", compute_accuracy(train_Y[:10, 0].to(DEVICE), torch.argmax(model(train_X[:10].to(DEVICE))[0], dim=-1)))

    """ torchvision.transforms.functional.to_pil_image(input[0][3]).show()
    M1 = compress.random_bernoulli_matrix((100, 224))
    M2 = compress.random_bernoulli_matrix((100, 224))
    compressed_clip = compress.compress_tensor(input[0], [M1, M2], [3, 2])
    torchvision.transforms.functional.to_pil_image(compressed_clip[3]).show()
    input[0] = compress.expand_tensor(compressed_clip, [M1.T, M2.T], [3, 2])
    torchvision.transforms.functional.to_pil_image(clips[0][3]).show() """

if __name__ == "__main__":
    main(parser.parse_args())
