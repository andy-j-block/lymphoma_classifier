import streamlit as st
import os
import pickle
from ImageLoader import ImageLoader
import pandas as pd
import random
from PIL import Image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
from typing import Tuple
from PytorchAlgos import PytorchAlgos
import torch


with open('streamlit_seed.obj', 'rb') as f:
    SEED_VAL = pickle.load(f)


def set_new_seed(CURRENT_SEED):
    CURRENT_SEED += 1
    with open('streamlit_seed.obj', 'wb') as f:
        pickle.dump(CURRENT_SEED, f)


class ImageData:

    def __init__(self):
        if 'image_loader.obj' in os.listdir(os.getcwd()):
            with open('image_loader.obj', 'rb') as f:
                self.image_df= pickle.load(f)

        else:
            self.image_df = ImageLoader('./Images').df

        decoder = {i: cancer_type for (i, cancer_type) in enumerate(['CLL', 'FL', 'MCL'])}
        self.image_df['cancer_type'] = self.image_df['cancer_type'].map(decoder)
        self.seed = SEED_VAL
        random.seed(self.seed)

    def random_sample(self):
        random_idx = random.randint(0, len(self.image_df))
        random_image = self.image_df['img_array'].iloc[random_idx]
        random_cancer_type = self.image_df['cancer_type'].iloc[random_idx]

        return random_image, random_cancer_type


class AlbTrxs:

    def __init__(self, image_data: ImageData, horizontal_flip: bool, vertical_flip: bool, color_jitter: bool, rotate: bool, rgb_shift: bool):
        self.image_data = ImageData
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.color_jitter = color_jitter
        self.rotate = rotate
        self.rgb_shift = rgb_shift

    def transform(self):
        A.Compose([A.HorizontalFlip(p=self.horizontal_flip),
                   A.VerticalFlip(p=self.vertical_flip),
                   A.ColorJitter(p=self.color_jitter),
                   A.Rotate(limit=10, interpolation=cv2.BORDER_CONSTANT, p=self.rotate),
                   A.RGBShift(p=self.rgb_shift)
                   ])


class TrainedModel:
    resize_factor: int = 347  # each image is 1388x1040 pixels, so this value represents a 4x downsize
    tl_means: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    tl_stds: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __init__(self):
        ### TODO fix the algorithm when model is selected
        pytorch_algos = PytorchAlgos()
        self.model = pytorch_algos.RESNET34.model
        self.model.load_state_dict(torch.load('./Model_State_Dicts/RESNET34/state_dict_1.pth'))

    def transform(self):
        A.Compose([A.Normalize(mean=self.tl_means, std=self.tl_stds),
                   A.LongestMaxSize(self.resize_factor),
                   ToTensorV2()
                   ])


header = st.beta_container()
refresh = st.beta_container()
image = st.beta_container()
run_model = st.beta_container()

with header:
    st.title('Lymphoma Subtype Classifier')
    st.subheader('')

with refresh:
    _, button_col, show_seed = st.beta_columns(3)
    button = button_col.button('Get new sample', on_click=set_new_seed(SEED_VAL))
    button_col.subheader('')
    show_seed.write(f'{SEED_VAL}')

with image:
    image_data = ImageData()
    image, cancer_type = image_data.random_sample()
    image = Image.fromarray(image)

    image_col, selection_col = st.beta_columns([5, 2])
    image_col.image(image, use_column_width=True)

    selection_col.header('Albumentations')
    horizontal_flip = selection_col.checkbox('Horizontal Flip')
    vertical_flip = selection_col.checkbox('Vertical Flip')
    color_jitter = selection_col.checkbox('Color Jitter')
    rotate = selection_col.checkbox('Rotation')
    rgb_shift = selection_col.checkbox('RGB Shift')

    data_prepper = AlbTrxs(image_data, horizontal_flip, vertical_flip, color_jitter, rotate, rgb_shift)

    apply_albs = selection_col.button('Apply Albumentations')#, on_click=)


with run_model:

    run_model_col, prediction_col, actual_col = st.beta_columns(3)
    run_model_col.subheader('')
    run_model_button = run_model_col.button('Run Model')

    prediction_col.subheader('Predicted Type:')
    prediction_col.write(f'the prediction')

    actual_col.subheader('Actual Type:')
    actual_col.write(f'the actual')
