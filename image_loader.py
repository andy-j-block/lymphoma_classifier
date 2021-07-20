from PIL import Image
import os
import numpy as np

class image_loader:

    def get_cancer_types(self, top_img_dir: str):
        self.cancer_types = [cancer_type for cancer_type in os.listdir(top_img_dir)]
        self.top_img_dir = top_img_dir


    def get_img_dirs(self):
        # get directories where images are stored
        self.img_dirs = [os.path.join(self.top_img_dir, cancer_type) for cancer_type in self.cancer_types]


    def load_images(self):
        # read images into a list
        self.imgs = []

        for i, img_dir in enumerate(self.img_dirs):
            img_paths = os.listdir(img_dir)

            for j in img_paths:
                # pass thru all the image files per image directory
                # open the image with pillow, convert to numpy array, add it to the images list with its cancer type label
                img_path = os.path.join(self.img_dirs[i], j)
                img_array = Image.open(img_path)
                img_array = np.asarray(img_array)
                self.imgs.append((cancer_types[i], img_array))