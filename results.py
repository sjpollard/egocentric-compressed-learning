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
    "--plot-results-bar", 
    action="store_true", 
    help="Plots spatial W&B results as bar chart"
)
parser.add_argument(
    "--plot-results-scatter", 
    action="store_true", 
    help="Plots all W&B results as scatter chart"
)
parser.add_argument(
    "--table-results", 
    action="store_true", 
    help="Saves results to a csv"
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

def plot_bar_graphs():
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
    plot_accuracies_bar("Verb Accuracy - Spatial Compression - Test Split",
                    'verb_spatial_test',
                    [1, 0.5, 0.25, 0.1, 0.01], 
                    spatial_results,
                    target='verb')
    plot_accuracies_bar("Noun Accuracy - Spatial Compression - Test Split",
                    'noun_spatial_test',
                    [1, 0.5, 0.25, 0.1, 0.01], 
                    spatial_results,
                    target='noun')

def plot_accuracies_bar(title, filename, measurement_rates, results_files, target):
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
    plt.savefig(f'images/{filename}.png')
    plt.savefig(f'images/{filename}.pdf')
    plt.clf()

def plot_scatter_graphs():
    bernoulli_results = [['P01_P02_20_v1.csv'], 
                         ['P01_P02_bernoulli_158_158_2_3_20_v1.csv',
                         'P01_P02_bernoulli_112_112_2_3_20_v1.csv',
                         'P01_P02_bernoulli_71_71_2_3_20_v1.csv',
                         'P01_P02_bernoulli_22_22_2_3_20_v1.csv'],
                         ['P01_P02_bernoulli_1_1_20_v1.csv'],
                         ['P01_P02_bernoulli_4_0_20_v1.csv',
                         'P01_P02_bernoulli_2_0_20_v1.csv',
                         'P01_P02_bernoulli_1_0_20_v1.csv']]
    gaussian_results = [['P01_P02_20_v1.csv'], 
                         ['P01_P02_gaussian_158_158_2_3_20_v1.csv',
                         'P01_P02_gaussian_112_112_2_3_20_v1.csv',
                         'P01_P02_gaussian_71_71_2_3_20_v1.csv',
                         'P01_P02_gaussian_22_22_2_3_20_v1.csv'],
                         ['P01_P02_gaussian_1_1_20_v1.csv'],
                         ['P01_P02_gaussian_4_0_20_v1.csv',
                         'P01_P02_gaussian_2_0_20_v1.csv',
                         'P01_P02_gaussian_1_0_20_v1.csv']]
    bernoulli_learnt_phi_theta_results = [['P01_P02_20_v1.csv'], 
                         ['P01_P02_bernoulli_learnt_phi_theta_158_158_2_3_20_v1.csv',
                         'P01_P02_bernoulli_learnt_phi_theta_112_112_2_3_20_v1.csv',
                         'P01_P02_bernoulli_learnt_phi_theta_71_71_2_3_20_v1.csv',
                         'P01_P02_bernoulli_learnt_phi_theta_22_22_2_3_20_v1.csv'],
                         ['P01_P02_bernoulli_learnt_phi_theta_1_1_20_v1.csv'],
                         ['P01_P02_bernoulli_learnt_phi_theta_4_0_20_v1.csv',
                         'P01_P02_bernoulli_learnt_phi_theta_2_0_20_v1.csv',
                         'P01_P02_bernoulli_learnt_phi_theta_1_0_20_v1.csv']]
    gaussian_learnt_phi_theta_results = [['P01_P02_20_v1.csv'], 
                         ['P01_P02_gaussian_learnt_phi_theta_158_158_2_3_20_v1.csv',
                         'P01_P02_gaussian_learnt_phi_theta_112_112_2_3_20_v1.csv',
                         'P01_P02_gaussian_learnt_phi_theta_71_71_2_3_20_v1.csv',
                         'P01_P02_gaussian_learnt_phi_theta_22_22_2_3_20_v1.csv'],
                         ['P01_P02_gaussian_learnt_phi_theta_1_1_20_v1.csv'],
                         ['P01_P02_gaussian_learnt_phi_theta_4_0_20_v1.csv',
                         'P01_P02_gaussian_learnt_phi_theta_2_0_20_v1.csv',
                         'P01_P02_gaussian_learnt_phi_theta_1_0_20_v1.csv']]
    measurement_rates = [[1], [0.5, 0.25, 0.1, 0.01], [0.33], [0.5, 0.25, 0.125]]
    colours = ['k', 'b', 'r', 'g']
    labels = ["Oracle", "Spatial", "Channel", "Temporal"]
    plot_accuracies_scatter("Verb Accuracy - Bernoulli", 'verb_bernoulli_all', measurement_rates, bernoulli_results, colours, labels, 'verb', 'test')
    plot_accuracies_scatter("Verb Accuracy - Gaussian", 'verb_gaussian_all', measurement_rates, gaussian_results, colours, labels, 'verb', 'test')
    plot_accuracies_scatter("Verb Accuracy - Bernoulli + Learnt", 'verb_bernoulli_learnt_all', measurement_rates, bernoulli_learnt_phi_theta_results, colours, labels, 'verb', 'test')
    plot_accuracies_scatter("Verb Accuracy - Gaussian + Learnt", 'verb_gaussian_learnt_all', measurement_rates, gaussian_learnt_phi_theta_results, colours, labels, 'verb', 'test')
    plot_accuracies_scatter("Noun Accuracy - Bernoulli", 'noun_bernoulli_all', measurement_rates, bernoulli_results, colours, labels, 'noun', 'test')
    plot_accuracies_scatter("Noun Accuracy - Gaussian", 'noun_gaussian_all', measurement_rates, gaussian_results, colours, labels, 'noun', 'test')
    plot_accuracies_scatter("Noun Accuracy - Bernoulli + Learnt", 'noun_bernoulli_learnt_all', measurement_rates, bernoulli_learnt_phi_theta_results, colours, labels, 'noun', 'test')
    plot_accuracies_scatter("Noun Accuracy - Gaussian + Learnt", 'noun_gaussian_learnt_all', measurement_rates, gaussian_learnt_phi_theta_results, colours, labels, 'noun', 'test')

def plot_accuracies_scatter(title, filename, measurement_rates, results_files, colours, labels, target, split):
    plt.title(title)
    plt.xlim([-0.02, 1.02])
    if target == 'verb':
        plt.ylim([0.25, 0.55])
    elif target == 'noun':
        plt.ylim([0.15, 0.60])
    for i in range(len(results_files)):
        accuracies = []
        for results_file in results_files[i]:
            results_df = pd.read_csv(f'results/{results_file}')
            accuracies.append(results_df[f'{split}/{target}-accuracy'].dropna().tail(3).mean())
        plt.scatter(measurement_rates[i], accuracies, c=colours[i], marker='x', label=labels[i])
    plt.legend(loc='lower right')
    plt.xlabel("Measurement rate")
    plt.ylabel("Accuracy(%)")
    plt.savefig(f'images/{filename}.png')
    plt.savefig(f'images/{filename}.pdf')
    plt.clf()

def save_tables():
    spatial_results = ['P01_P02_20_v1.csv', 
                       'P01_P02_bernoulli_158_158_2_3_20_v1.csv', 
                       'P01_P02_bernoulli_112_112_2_3_20_v1.csv',
                       'P01_P02_bernoulli_71_71_2_3_20_v1.csv',
                       'P01_P02_bernoulli_22_22_2_3_20_v1.csv',
                       'P01_P02_gaussian_158_158_2_3_20_v1.csv', 
                       'P01_P02_gaussian_112_112_2_3_20_v1.csv',
                       'P01_P02_gaussian_71_71_2_3_20_v1.csv',
                       'P01_P02_gaussian_22_22_2_3_20_v1.csv',
                       'P01_P02_bernoulli_learnt_phi_theta_158_158_2_3_20_v1.csv', 
                       'P01_P02_bernoulli_learnt_phi_theta_112_112_2_3_20_v1.csv',
                       'P01_P02_bernoulli_learnt_phi_theta_71_71_2_3_20_v1.csv',
                       'P01_P02_bernoulli_learnt_phi_theta_22_22_2_3_20_v1.csv',
                       'P01_P02_gaussian_learnt_phi_theta_158_158_2_3_20_v1.csv', 
                       'P01_P02_gaussian_learnt_phi_theta_112_112_2_3_20_v1.csv',
                       'P01_P02_gaussian_learnt_phi_theta_71_71_2_3_20_v1.csv',
                       'P01_P02_gaussian_learnt_phi_theta_22_22_2_3_20_v1.csv']
    channel_results = ['P01_P02_20_v1.csv', 
                       'P01_P02_bernoulli_1_1_20_v1.csv',
                       'P01_P02_gaussian_1_1_20_v1.csv',
                       'P01_P02_bernoulli_learnt_phi_theta_1_1_20_v1.csv',
                       'P01_P02_gaussian_learnt_phi_theta_1_1_20_v1.csv']
    temporal_results = ['P01_P02_20_v1.csv', 
                        'P01_P02_bernoulli_4_0_20_v1.csv', 
                        'P01_P02_bernoulli_2_0_20_v1.csv',
                        'P01_P02_bernoulli_1_0_20_v1.csv',
                        'P01_P02_gaussian_4_0_20_v1.csv', 
                        'P01_P02_gaussian_2_0_20_v1.csv',
                        'P01_P02_gaussian_1_0_20_v1.csv',
                        'P01_P02_bernoulli_learnt_phi_theta_4_0_20_v1.csv', 
                        'P01_P02_bernoulli_learnt_phi_theta_2_0_20_v1.csv',
                        'P01_P02_bernoulli_learnt_phi_theta_1_0_20_v1.csv',
                        'P01_P02_gaussian_learnt_phi_theta_4_0_20_v1.csv', 
                        'P01_P02_gaussian_learnt_phi_theta_2_0_20_v1.csv',
                        'P01_P02_gaussian_learnt_phi_theta_1_0_20_v1.csv']
    width_results = ['P01_P02_20_v1.csv',
                     'P01_P02_bernoulli_112_3_20_v1.csv',
                     'P01_P02_bernoulli_56_3_20_v1.csv',
                     'P01_P02_bernoulli_22_3_20_v1.csv',
                     'P01_P02_bernoulli_2_3_20_v1.csv']
    height_results = ['P01_P02_20_v1.csv',
                      'P01_P02_bernoulli_112_2_20_v1.csv',
                      'P01_P02_bernoulli_56_2_20_v1.csv',
                      'P01_P02_bernoulli_22_2_20_v1.csv',
                      'P01_P02_bernoulli_2_2_20_v1.csv']
    save_accuracies('verb_spatial_test', spatial_results, target='verb', split='test')
    save_accuracies('noun_spatial_test', spatial_results, target='noun', split='test')
    save_accuracies('verb_channel_test', channel_results, target='verb', split='test')
    save_accuracies('noun_channel_test', channel_results, target='noun', split='test')
    save_accuracies('verb_temporal_test', temporal_results, target='verb', split='test')
    save_accuracies('noun_temporal_test', temporal_results, target='noun', split='test')
    save_accuracies('verb_width_test', width_results, target='verb', split='test')
    save_accuracies('noun_width_test', width_results, target='noun', split='test')
    save_accuracies('verb_height_test', height_results, target='verb', split='test')
    save_accuracies('noun_height_test', height_results, target='noun', split='test')

def save_accuracies(filename, results_files, target, split):
    accuracies = []
    for results_file in results_files:
        results_df = pd.read_csv(f'results/{results_file}')
        accuracies.append(f"{results_df[f'{split}/{target}-accuracy'].dropna().tail(3).mean() * 100:.2f}")
    table_df = pd.DataFrame({'File': results_files, 'Accuracy': accuracies})
    table_df.to_csv(f'results/{filename}.csv', index=False)

def main(args):
    if args.pull_results:
        results_to_csv()
    if args.plot_results_bar:
        plot_bar_graphs()
    if args.plot_results_scatter:
        plot_scatter_graphs()
    if args.table_results:
        save_tables()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


if __name__ == "__main__":
    main(parser.parse_args())
