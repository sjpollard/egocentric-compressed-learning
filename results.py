import argparse

import pandas as pd 
import wandb
import os

parser = argparse.ArgumentParser(
    description="Hub for everything necessary to train the neural network",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--pull-results", 
    action="store_true", 
    help="Pulls results from wandb into csv"
)

def results_to_csv():
    if not os.path.exists(f'results'):
        os.makedirs(f'results')

    api = wandb.Api()
    entity, project = 'sjpollard', 'egocentric-compressed-learning-results'
    runs = api.runs(entity + "/" + project)

    for run in runs:
        name = run.config['model_label']
        history_df = run.history(samples=run.summary['_step'] + 1)
        run_df = history_df[['train/verb-accuracy', 'train/noun-accuracy', 'val/verb-accuracy', 'val/noun-accuracy', 'test/verb-accuracy', 'test/noun-accuracy']]
        run_df.to_csv(f'results/{name}.csv', index=False)

def main(args):
    if args.pull_results:
        results_to_csv()

if __name__ == "__main__":
    main(parser.parse_args())
