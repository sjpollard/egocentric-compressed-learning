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

def plot_accuracies(title, filename, measurement_rates, results_files, target):
    plt.title(title)
    plt.xticks(measurement_rates)
    width = 0.2
    shift = 0
    plt.ylim([0, 0.7])
    labels = ["Bernoulli", "Gaussian", "Learnt Bernoulli", "Learnt Gaussian"]
    for i in range(len(results_files)):
        accuracies = []
        for results_file in results_files[i]:
            results_df = pd.read_csv(f'results/{results_file}')
            accuracies.append(results_df[f'test/{target}-accuracy'].dropna().tail(3).mean())
        plt.bar(np.arange(len(measurement_rates)) + shift, height=accuracies, width=width, label=labels[i])
        shift += width
    plt.xticks(np.arange(len(measurement_rates)) + shift / (len(labels) - 1), measurement_rates)
    plt.legend(loc="best")
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
                         'P01_P02_gaussian_22_22_2_3_20_v1.csv'],
                         ['P01_P02_20_v1.csv', 
                         'P01_P02_bernoulli_learnt_phi_theta_158_158_2_3_20_v1.csv', 
                         'P01_P02_bernoulli_learnt_phi_theta_112_112_2_3_20_v1.csv',
                         'P01_P02_bernoulli_learnt_phi_theta_71_71_2_3_20_v1.csv',
                         'P01_P02_bernoulli_learnt_phi_theta_22_22_2_3_20_v1.csv'],
                         ['P01_P02_20_v1.csv', 
                         'P01_P02_gaussian_learnt_phi_theta_158_158_2_3_20_v1.csv', 
                         'P01_P02_gaussian_learnt_phi_theta_112_112_2_3_20_v1.csv',
                         'P01_P02_gaussian_learnt_phi_theta_71_71_2_3_20_v1.csv',
                         'P01_P02_gaussian_learnt_phi_theta_22_22_2_3_20_v1.csv']]
        plot_accuracies("Verb Accuracy - Spatial Compression - Test Split",
                        'verb_spatial_test',
                        [1, 0.5, 0.25, 0.1, 0.01], 
                        spatial_results,
                        target='verb')
        plot_accuracies("Noun Accuracy - Spatial Compression - Test Split",
                        'noun_spatial_test',
                        [1, 0.5, 0.25, 0.1, 0.01], 
                        spatial_results,
                        target='noun')


if __name__ == "__main__":
    main(parser.parse_args())
