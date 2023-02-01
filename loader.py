import numpy as np
import tensorly as tl
from PIL import Image

tl.set_backend('numpy')


def get_annotation_frames(annotation):
    participant_id = annotation[0]
    video_id = annotation[1]
    start_frame = annotation[2]
    stop_frame = annotation[3]
    zero = '0'
    i = map(str, range(start_frame, stop_frame + 1))
    file_names = list(map(lambda x: f'{(10 - len(x)) * zero}{x}', i))
    frames = list(map(lambda x: np.asarray(Image.open(
        f'EPIC-KITCHENS/{participant_id}/rgb_frames/{video_id}/frame_{x}.jpg')), file_names))
    return tl.tensor(frames), (annotation[4], annotation[5])


def get_annotation_snippets(annotation, num_segments):
    participant_id = annotation[0]
    video_id = annotation[1]
    start_frame = annotation[2]
    stop_frame = annotation[3]
    zero = '0'
    segments = np.array_split(
        np.arange(start_frame, stop_frame + 1), num_segments)
    snippets = list(
        map(lambda x: str(np.random.default_rng().choice(x)), segments))
    file_names = list(map(lambda x: f'{(10 - len(x)) * zero}{x}', snippets))
    frames = list(map(lambda x: np.asarray(Image.open(
        f'EPIC-KITCHENS/{participant_id}/rgb_frames/{video_id}/frame_{x}.jpg')), file_names))
    return tl.tensor(frames), (annotation[4], annotation[5])
