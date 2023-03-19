import argparse

import pandas as pd 
import wandb
import os

parser = argparse.ArgumentParser(
    description="Manages compilation and plotting of results from W&B",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--pull-results", 
    action="store_true", 
    help="Pulls results from W&B into csv"
)
parser.add_argument(
    "--plot-results", 
    action="store_true", 
    help="Plots W&B results"
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

def plot_accuracies():
    return

def main(args):
    if args.pull_results:
        results_to_csv()
    if args.plot_results:
        val_accuracies = []
        test_accuracies = []
        results_files = ['P01_P02_20_v1.csv', 
                         'P01_P02_bernoulli_158_158_2_3_20_v1.csv', 
                         'P01_P02_bernoulli_112_112_2_3_20_v1.csv',
                         'P01_P02_bernoulli_71_71_2_3_20_v1.csv',
                         'P01_P02_bernoulli_22_22_2_3_20_v1.csv']
        for results_file in results_files:
            results_df = pd.read_csv(f'results/{results_file}')
            val_accuracies.append([results_df['val/verb-accuracy'].dropna().tail(3).mean(), 
                                   results_df['val/noun-accuracy'].dropna().tail(3).mean(),])
            test_accuracies.append([results_df['test/verb-accuracy'].dropna().tail(3).mean(), 
                                    results_df['test/noun-accuracy'].dropna().tail(3).mean(),])
        print(val_accuracies)
        print(test_accuracies)


if __name__ == "__main__":
    main(parser.parse_args())
