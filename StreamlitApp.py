import streamlit as st
import os
import pickle
from ImageLoader import ImageLoader
import pandas as pd
import random
from PIL import Image


class ImageData:

    def __init__(self):
        if 'image_loader.obj' in os.listdir(os.getcwd()):
            with open('image_loader.obj', 'rb') as f:
                self.image_df= pickle.load(f)

        else:
            self.image_df = ImageLoader('./Images').df

        decoder = {i: cancer_type for (i, cancer_type) in enumerate(['CLL', 'FL', 'MCL'])}
        self.image_df['cancer_type'] = self.image_df['cancer_type'].map(decoder)

    def random_sample(self):
        random_idx = random.randint(0, len(self.image_df))
        random_image = self.image_df['img_array'].iloc[random_idx]
        random_cancer_type = self.image_df['cancer_type'].iloc[random_idx]

        return random_image, random_cancer_type


header = st.beta_container()
image = st.beta_container()
augmentations = st.beta_container()
analyze = st.beta_container()

with header:
    st.title('Lymphoma Subtype Classifier')

with image:
    image_data = ImageData()
    image, cancer_type = image_data.random_sample()
    image = Image.fromarray(image)

    st.header(f'Cancer type: {cancer_type}')
    st.image(image, use_column_width=True)


with analyze:
    algo_select = st.radio('Choose a trained model:', ('Resnet', 'VGG', 'VGG_BN'))

    run_model = st.button('Analyze sample')


# with augmentations:

    # horizontal_flip = st.slider()
    # vertical_flip = st.slider()
    # color_jitter = st.slider()
    # rotate = st.slider()
    # rgb_shift = st.slider()

    # A.HorizontalFlip(p=self.p),
    # A.VerticalFlip(p=self.p),
    # A.ColorJitter(p=self.p),
    # A.Rotate(limit=10, interpolation=cv2.BORDER_CONSTANT, p=self.p),
    # A.RGBShift(p=self.p)