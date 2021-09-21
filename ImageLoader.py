from PIL import Image
import os
import numpy as np
import random
import pandas as pd
import pickle


def label_encoder():
    return {cancer_type: enum for (enum, cancer_type) in enumerate(['CLL', 'FL', 'MCL'])}


def label_decoder():
    return {enum: cancer_type for (enum, cancer_type) in enumerate(['CLL', 'FL', 'MCL'])}


class ImageLoader:
    def __init__(self, top_img_dir: str):
        self.top_img_dir = top_img_dir
        """get the cancer types from the subdirectory names and their full paths"""
        self.cancer_types = [cancer_type for cancer_type in os.listdir(self.top_img_dir)]
        self.img_dirs = [os.path.join(self.top_img_dir, cancer_type) for cancer_type in self.cancer_types]

        """load the images with PIL/numpy and save them to a pandas dataframe"""
        self.imgs_and_labels = self.load_images()
        self.df = pd.DataFrame(self.imgs_and_labels, columns=['cancer_type', 'img_array'])

        """add label encoding because pytorch doesnt handle strings"""
        self.df['cancer_type'] = self.df['cancer_type'].map(label_encoder())
        self.overfit_df = self.df.iloc[random.sample(range(0, len(self.df)), 4)]


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


def main(top_img_dir: str = './Images', pickle_=True):

    image_loader = ImageLoader(top_img_dir=top_img_dir)

    # if pickle_:
    #     with open('image_loader.obj', 'wb') as f:
    #         pickle.dump(image_loader.df, f)
    #         print('ImageLoader object saved successfully')
    #
    #     with open('overfit_data.obj', 'wb') as f:
    #         pickle.dump(image_loader.overfit_df, f)
    #         print('Overfit test data object saved successfully')


if __name__ == '__main__':
    main()
