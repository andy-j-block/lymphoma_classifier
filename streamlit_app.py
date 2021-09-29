import numpy as np
import pandas as pd
import streamlit as st
import os
import pickle
from ImageLoader import ImageLoader, label_encoder, label_decoder
from PIL import Image
from PytorchAlgos import PytorchAlgos
from KFolder import *
# import urllib.request


# @st.cache
# def download1(url1):
#     url = url1
#     filename = url.split('/')[-1]
#     urllib.request.urlretrieve(url, filename)
#
#
# download1('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')


# Initialization
TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if 'SEED_VAL' not in st.session_state:
    st.session_state['SEED_VAL'] = 0

ALB_TRX_STATES = ['hflip', 'vflip', 'color_jitter', 'rotate', 'rgb_shift']
for trx_state in ALB_TRX_STATES:
    if trx_state not in st.session_state:
        st.session_state[trx_state] = False


RESIZE_FACTOR: int = 347  # each image is 1388x1040 pixels, so this value represents a 4x downsize
MEANS: Tuple[float, float, float] = (0.485, 0.456, 0.406)
STDS: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@st.experimental_singleton
def get_image_dataframe() -> pd.DataFrame:
    if 'image_loader.obj' in os.listdir(os.getcwd()):
        with open('image_loader.obj', 'rb') as f:
            return pickle.load(f)
    else:
        return ImageLoader('./Images').df


def get_random_sample(image_df: pd.DataFrame) -> Tuple[np.ndarray, np.int64]:
    random.seed(st.session_state['SEED_VAL'])
    random_idx = random.randint(0, len(image_df))
    return image_df['img_array'].iloc[random_idx], image_df['cancer_type'].iloc[random_idx]


@st.experimental_memo
def apply_alb_trxs(image_: np.ndarray, horizontal_flip: bool, vertical_flip: bool, color_jitter: bool,
                   rotate: bool, rgb_shift: bool) -> np.ndarray:

    return A.Compose([A.HorizontalFlip(p=horizontal_flip),
                      A.VerticalFlip(p=vertical_flip),
                      A.ColorJitter(p=color_jitter),
                      A.Rotate(limit=10, interpolation=cv2.BORDER_CONSTANT, p=rotate),
                      A.RGBShift(p=rgb_shift)
                      ])(image=image_)['image']


@st.experimental_singleton
def create_model(torch_device: torch.device):
    pytorch_algos = PytorchAlgos(resnet101=True, resnet18=False, resnet34=False, vgg13=False, vgg16=False,
                                 vgg13_bn=False, vgg16_bn=False, mobilenet_v2=False)
    model = pytorch_algos.RESNET101.model
    model.load_state_dict(torch.load('./best_model.pth', map_location=torch_device))
    return model


@st.experimental_memo
def pytoch_default_transforms(image_: np.ndarray, means: Tuple[float, float, float],
                              stds: Tuple[float, float, float], resize_factor: int) -> np.ndarray:
    return A.Compose([A.Normalize(mean=means, std=stds),
                      A.LongestMaxSize(resize_factor),
                      ToTensorV2()
                      ])(image=image_)['image']


@st.experimental_memo
def make_prediction(_model_, _image_: np.ndarray, cancer_type: int, torch_device: torch.device):
    data = {'cancer_type': [cancer_type], 'img_array': [_image_]}
    input_df = pd.DataFrame(data=data)

    dataset = PytorchImagesDataset(input_df)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(torch_device), labels.to(torch_device)

        with torch.set_grad_enabled(False):
            outputs = _model_(inputs)
            _, preds = torch.max(outputs, 1)

        decoder = label_decoder()
        prediction = int(preds)
        prediction = decoder[prediction]
        cancer_type = decoder[cancer_type]

    return cancer_type, prediction


header = st.container()
image = st.container()
run_model = st.container()


with header:
    st.title('Lymphoma Subtype Classifier')
    st.write("""
    This app uses a trained model to classify sample Non-Hodgkins Lymphoma biopsy images into three sub-types: \
    Chronic Lymphocytic Leukemia (CLL), Follicular Lymphoma (FL), or Mantle Cell Lymphoma (MCL).  Toggle the \
    "Transformations" buttons to alter the image as you like before clicking the "Run Model" button, which returns \
    the predicted and actual types.
    """)
    st.write('')


IMAGE: np.ndarray
CANCER_TYPE: np.int64

with image:

    # define layout and add interactive elements
    image_col, selection_col = st.columns([5, 2])

    selection_col.header('Transformations')
    horizontal_flip = selection_col.checkbox('Horizontal Flip')
    vertical_flip = selection_col.checkbox('Vertical Flip')
    color_jitter = selection_col.checkbox('Color Jitter')
    rotate = selection_col.checkbox('Rotation')
    rgb_shift = selection_col.checkbox('RGB Shift')

    # define transformations and their state sessions
    button_objs = [horizontal_flip, vertical_flip, color_jitter, rotate, rgb_shift]
    session_states = []
    for i, trx_state in enumerate(ALB_TRX_STATES):
        st.session_state[trx_state] = True if button_objs[i] else False
        session_states.append(st.session_state[trx_state])

    # get new sample button changes seed value on change
    selection_col.write('')
    button = selection_col.button('Get new sample')
    if button:
        st.session_state['SEED_VAL'] += 1

    # albumentations transformations and image display
    image_data = get_image_dataframe()
    IMAGE, CANCER_TYPE = get_random_sample(image_data)
    IMAGE = apply_alb_trxs(IMAGE, *session_states)
    image_col.image(Image.fromarray(IMAGE), use_column_width=True)


with run_model:
    model = create_model(TORCH_DEVICE)
    IMAGE = pytoch_default_transforms(IMAGE, MEANS, STDS, RESIZE_FACTOR)

    run_model_col, prediction_col, actual_col = st.columns(3)
    prediction_col.subheader('Predicted Type:')
    actual_col.subheader('Actual Type:')

    run_model_col.subheader('')
    run_model_button = run_model_col.button('Run Model')
    if run_model_button:
        actual, prediction = make_prediction(model, IMAGE, CANCER_TYPE, TORCH_DEVICE)
        prediction_col.write(prediction)
        actual_col.write(actual)
    else:
        prediction_col.write('Waiting')
        actual_col.write('Waiting')
