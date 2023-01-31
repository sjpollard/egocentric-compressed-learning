import compress
import numpy as np
import tensorly as tl
import pandas as pd
from PIL import Image

tl.set_backend('numpy')

def get_frames(annotation):
    participant_id = annotation[0]
    video_id = annotation[1]
    start_frame = annotation[2]
    stop_frame = annotation[3]
    zero = '0'
    for i in range(start_frame, stop_frame + 1):
        frame_id = f'{(10 - len(str(i))) * zero}{str(i)}'
        frame = Image.open(f'EPIC-KITCHENS/{participant_id}/rgb_frames/{video_id}/frame_{frame_id}.jpg')

def main():
    train_annotations = pd.read_csv('annotations\EPIC_100_train.csv')
    for annotation in train_annotations.loc[0:328, ['participant_id', 'video_id', 'start_frame', 'stop_frame', 'verb', 'noun']].values:
        print(annotation)
        get_frames(annotation)
    


if __name__ == '__main__':
    main()