import argparse

import pandas as pd 
import wandb
import os
import numpy as np

from matplotlib import pyplot as plt

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

def plot_verb_accuracies(title, filename, measurement_rates, results_files):
    bernoulli_test_accuracies = []
    gaussian_test_accuracies = []
    for results_file in results_files[0]:
        results_df = pd.read_csv(f'results/{results_file}')
        bernoulli_test_accuracies.append(results_df['test/verb-accuracy'].dropna().tail(3).mean())
    for results_file in results_files[1]:
        results_df = pd.read_csv(f'results/{results_file}')
        gaussian_test_accuracies.append(results_df['test/verb-accuracy'].dropna().tail(3).mean())
    plt.title(title)
    plt.xticks(measurement_rates)
    #plt.xlim([1, 8])
    #plt.ylim([2, 3.4])
    plt.plot(measurement_rates, bernoulli_test_accuracies, marker='x', label="Bernoulli")
    plt.plot(measurement_rates, gaussian_test_accuracies, marker='x', label="Gaussian")
    plt.legend(loc="upper left")
    plt.xlabel("Measurement rate")
    plt.ylabel("Accuracy(%)")
    plt.savefig(f'images/{filename}.pdf')
    plt.clf()

def plot_noun_accuracies(title, filename, measurement_rates, results_files):
    bernoulli_test_accuracies = []
    gaussian_test_accuracies = []
    for results_file in results_files[0]:
        results_df = pd.read_csv(f'results/{results_file}')
        bernoulli_test_accuracies.append(results_df['test/noun-accuracy'].dropna().tail(3).mean())
    for results_file in results_files[1]:
        results_df = pd.read_csv(f'results/{results_file}')
        gaussian_test_accuracies.append(results_df['test/noun-accuracy'].dropna().tail(3).mean())
    plt.title(title)
    plt.xticks(measurement_rates)
    #plt.xlim([1, 8])
    #plt.ylim([2, 3.4])
    plt.plot(measurement_rates, bernoulli_test_accuracies, marker='x', label="Bernoulli")
    plt.plot(measurement_rates, gaussian_test_accuracies, marker='x', label="Gaussian")
    plt.legend(loc="upper left")
    plt.xlabel("Measurement rate")
    plt.ylabel("Accuracy(%)")
    plt.savefig(f'images/{filename}.pdf')
    plt.clf()

def main(args):
    if args.pull_results:
        results_to_csv()
    if args.plot_results:
        spatial_results = [['P01_P02_20_v1.csv', 
                         'P01_P02_bernoulli_158_158_2_3_20_v1.csv', 
                         'P01_P02_bernoulli_112_112_2_3_20_v1.csv',
                         'P01_P02_bernoulli_71_71_2_3_20_v1.csv',
                         'P01_P02_bernoulli_22_22_2_3_20_v1.csv'],
                         ['P01_P02_20_v1.csv', 
                         'P01_P02_gaussian_158_158_2_3_20_v1.csv', 
                         'P01_P02_gaussian_112_112_2_3_20_v1.csv',
                         'P01_P02_gaussian_71_71_2_3_20_v1.csv',
                         'P01_P02_gaussian_22_22_2_3_20_v1.csv']]
        plot_verb_accuracies("Verb Accuracy - Spatial Compression - Test Split",
                        'verb_spatial_test',
                        [1, 0.5, 0.25, 0.1, 0.01], 
                        spatial_results)
        plot_noun_accuracies("Noun Accuracy - Spatial Compression - Test Split",
                        'noun_spatial_test',
                        [1, 0.5, 0.25, 0.1, 0.01], 
                        spatial_results)


if __name__ == "__main__":
    main(parser.parse_args())
