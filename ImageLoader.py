from PIL import Image
import os
import numpy as np
import random
import pandas as pd
import pickle


class ImageLoader:
    def __init__(self, top_img_dir: str):
        self.top_img_dir = top_img_dir
        self.cancer_types = [cancer_type for cancer_type in os.listdir(self.top_img_dir)]
        self.img_dirs = [os.path.join(self.top_img_dir, cancer_type) for cancer_type in self.cancer_types]

        self.imgs_and_labels = self.load_images()
        self.df = self.create_dataframe()

    def get_img_dirs(self):
        """get directories where images are stored"""
        return

    def load_images(self):
        """read images into a list"""
        imgs_and_labels = []

        for i, img_dir in enumerate(self.img_dirs):
            img_paths = os.listdir(img_dir)
            for j in img_paths:
                """pass thru all the image files per image directory, open the image with pillow, 
                convert to numpy array, add it to the images list with its cancer type label"""
                img_path = os.path.join(self.img_dirs[i], j)
                img_array = np.asarray(Image.open(img_path))
                imgs_and_labels.append((self.cancer_types[i], img_array))
        return imgs_and_labels

    ###TODO fix context manager to print a random data preview to file
    def data_preview(self):
        """print out some data entries"""
        random_data_entry = random.randint(0, len(self.imgs_and_labels))

        with open('data_preview', 'wb') as f:
            f.write(f'Random entry cancer type: {self.imgs_and_labels[random_data_entry][0]}\n')
            f.write(f'Random entry image array: {self.imgs_and_labels[random_data_entry][1][:3]}\ncontinued...\n')
            f.write('Random entry image:')
            Image.fromarray(self.imgs_and_labels[random_data_entry][1]).reduce(2)

        print(self.df.head(3))
        print(f'Images dataframe shape: {self.df.shape}\n')

    def create_dataframe(self):
        return pd.DataFrame(self.imgs_and_labels, columns=['cancer_type', 'img_array'])


def main(top_img_dir: str = './Images', pickle_=True):

    image_loader = ImageLoader(top_img_dir=top_img_dir)

    if pickle_:
        with open('image_loader.obj', 'wb') as f:
            pickle.dump(image_loader, f)
            print('ImageLoader object saved successfully')


if __name__ == '__main__':
    main()
