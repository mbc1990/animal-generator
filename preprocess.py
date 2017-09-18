import os

from keras.preprocessing.image import load_img

def resize_all(in_dir, out_dir, size):
    """
    Resizes all the images in the directory

    size (x, y) tuple of new size
    """
    fnames = os.listdir(in_dir)
    for fname in fnames:
        img = load_img(in_dir+fname)
        img = img.resize(size)
        img.save(out_dir+fname)


def main():
    in_dir = "goldens_filtered_150x150/"
    out_dir = "goldens_filtered_150x150/"
    size = (150, 150)
    resize_all(in_dir, out_dir, size)
    
if __name__ == "__main__":
    main()
