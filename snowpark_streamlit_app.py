import numpy as np
import pandas as pd
import streamlit as st
import io
import os
from utils import label_decoder, label_encoder
from ImageLoader_snowpark import ImageLoader, get_all_img_ids
from PIL import Image
from PytorchAlgos import PytorchAlgos
from KFolder import *

from snowflake.snowpark import Session
from snowflake_credentials import get_credentials

# st.cache_data.clear()

credentials = get_credentials()
session = Session.builder.configs(credentials).create()
# images = ImageLoader(session)
img_ids = get_all_img_ids(session, 'all_images')

ENCODER = label_encoder()
DECODER = label_decoder()


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


# @st.cache_data
def get_image_dataframe(img_loader: ImageLoader) -> pd.DataFrame:
    # if 'image_loader.obj' in os.listdir(os.getcwd()):
    #     with open('image_loader.obj', 'rb') as f:
    #         return pickle.load(f)
    # else:
    #     return ImageLoader('./Images').df
    return img_loader.load_images()

# @st.cache_data
def get_random_sample(_session: Session) -> Tuple[np.ndarray, np.int64]:
    random.seed(st.session_state['SEED_VAL'])
    random_idx = random.choice(img_ids)
    
    img_bytes_and_ctype = _session.sql(f"select img_bytes, cancer_type from all_images where id='{random_idx}';").collect()
    
    ctype = img_bytes_and_ctype[0]['CANCER_TYPE']
    ctype_encoded = ENCODER[ctype]

    stage = f'raw_images_{ctype}'
    random_idx += '.tif'

    temp_dir = 'Images/temp'
    session.file.get(f"@{stage}/{random_idx}", temp_dir)

    file_path = os.path.join(temp_dir, random_idx)

    with open(file_path, 'rb') as f:
        img = Image.open(f)
        img_array = np.asarray(img)

    return img_array, ctype_encoded


@st.cache_data
def apply_alb_trxs(image_: np.ndarray, horizontal_flip: bool, vertical_flip: bool, color_jitter: bool,
                   rotate: bool, rgb_shift: bool) -> np.ndarray:

    return A.Compose([A.HorizontalFlip(p=horizontal_flip),
                      A.VerticalFlip(p=vertical_flip),
                      A.ColorJitter(p=color_jitter),
                      A.Rotate(limit=10, interpolation=cv2.BORDER_CONSTANT, p=rotate),
                      A.RGBShift(p=rgb_shift)
                      ])(image=image_)['image']


@st.cache_resource
def create_model(torch_device: torch.device):
    pytorch_algos = PytorchAlgos(resnet101=True, resnet18=False, resnet34=False, vgg13=False, vgg16=False,
                                 vgg13_bn=False, vgg16_bn=False, mobilenet_v2=False)
    model_ = pytorch_algos.RESNET101.model
    model_.load_state_dict(torch.load('./best_model.pth', map_location=torch_device))
    return model_


@st.cache_data
def pytorch_default_transforms(image_: np.ndarray, means: Tuple[float, float, float],
                               stds: Tuple[float, float, float], resize_factor: int) -> np.ndarray:
    return A.Compose([A.Normalize(mean=means, std=stds),
                      A.LongestMaxSize(resize_factor),
                      ToTensorV2()
                      ])(image=image_)['image']


# @st.cache_data
def make_prediction(_model_, _image_: np.ndarray, cancer_type: int, torch_device: torch.device) -> Tuple[str, str]:
    data = {'cancer_type': [cancer_type], 'img_array': [_image_]}
    input_df = pd.DataFrame(data=data)

    dataset = PytorchImagesDataset(input_df)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    prediction = ''
    actual = ''

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(torch_device), labels.to(torch_device)

        with torch.set_grad_enabled(False):
            outputs = _model_(inputs)
            _, preds = torch.max(outputs, 1)
            st.write(outputs)

        prediction = DECODER[int(preds)]
        actual = DECODER[cancer_type]

    return actual, prediction


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
    # image_data = get_image_dataframe(images)
    IMAGE_ARRAY, CANCER_TYPE = get_random_sample(session)
    IMAGE_ARRAY = apply_alb_trxs(IMAGE_ARRAY, *session_states)
    image_col.image(IMAGE_ARRAY, use_column_width=True)


with run_model:
    model = create_model(TORCH_DEVICE)
    IMAGE_ARRAY = pytorch_default_transforms(IMAGE_ARRAY, MEANS, STDS, RESIZE_FACTOR)

    run_model_col, prediction_col, actual_col = st.columns(3)
    prediction_col.subheader('Predicted Type:')
    actual_col.subheader('Actual Type:')

    run_model_col.subheader('')
    run_model_button = run_model_col.button('Run Model')
    if run_model_button:
        actual, prediction = make_prediction(model, IMAGE_ARRAY, CANCER_TYPE, TORCH_DEVICE)
        prediction_col.write(prediction)
        actual_col.write(actual)
    else:
        prediction_col.write('Waiting')
        actual_col.write('Waiting')

