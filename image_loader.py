from PIL import Image
import os
import numpy as np
import random
import pandas as pd
import pickle

class image_loader:

    def get_cancer_types(self, top_img_dir: str):
        self.cancer_types = [cancer_type for cancer_type in os.listdir(top_img_dir)]
        self.top_img_dir = top_img_dir


    def get_img_dirs(self):
        """get directories where images are stored"""
        self.img_dirs = [os.path.join(self.top_img_dir, cancer_type) for cancer_type in self.cancer_types]


    def load_images(self):
        """read images into a list"""

        self.imgs_and_labels = []
        for i, img_dir in enumerate(self.img_dirs):
            img_paths = os.listdir(img_dir)

            for j in img_paths:
                """pass thru all the image files per image directory, open the image with pillow, 
                convert to numpy array, add it to the images list with its cancer type label"""
                img_path = os.path.join(self.img_dirs[i], j)
                img_array = Image.open(img_path)
                img_array = np.asarray(img_array)
                self.imgs_and_labels.append((self.cancer_types[i], img_array))


    ###TODO fix context manager to print a random data preview to file
    def data_preview(self):
        """print out some data entries"""
        random_data_entry = random.randint(0, len(self.imgs_and_labels))

        with open('data_preview', 'wb') as f:
            f.write(f'Random entry cancer type: {self.imgs_and_labels[random_data_entry][0]}\n')
            f.write(f'Random entry image array: {self.imgs_and_labels[random_data_entry][1][:3]}\ncontinued...\n')
            f.write('Random entry image:')
            Image.fromarray(self.imgs_and_labels[random_data_entry][1]).reduce(2)


    def create_dataframe(self):
        self.df = pd.DataFrame(self.imgs_and_labels, columns=['cancer_type', 'img_array'])
        print(self.df.head(3))
        print(f'Images dataframe shape: {self.df.shape}\n')


images = image_loader()
images.get_cancer_types('./Images')
images.get_img_dirs()
images.load_images()
# images.data_preview()
images.create_dataframe()

with open('image_loader.obj', 'wb') as f:
    pickle.dump(images, f)