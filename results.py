import pandas as pd 
import wandb
import os

def main():

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

if __name__ == "__main__":
    main()
