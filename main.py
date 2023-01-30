import compress
import numpy as np
import tensorly as tl
from PIL import Image

tl.set_backend('numpy')

def main():
    W1 = compress.random_gaussian_matrix((1, 3))
    image = Image.open('EPIC-KITCHENS/P01/rgb_frames/P01_01/frame_0000000001.jpg')
    frame = np.asarray(image)
    print(compress.compress_tensor(frame, [W1], [2]).shape)


if __name__ == '__main__':
    main()