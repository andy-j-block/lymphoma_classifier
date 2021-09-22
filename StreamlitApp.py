import streamlit as st
import os
import pickle
from ImageLoader import ImageLoader, label_decoder
from PIL import Image
from PytorchAlgos import PytorchAlgos
from KFolder import *


# Initialization
TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if 'SEED_VAL' not in st.session_state:
    st.session_state['SEED_VAL'] = 42

ALB_TRX_STATES = ['hflip', 'vflip', 'color_jitter', 'rotate', 'rgb_shift']
for trx_state in ALB_TRX_STATES:
    if trx_state not in st.session_state:
        st.session_state[trx_state] = False

if 'apply_alb_trxs' not in st.session_state:
    st.session_state['apply_alb_trxs'] = False


class ImageData:
    transformed_image: Any
    PIL_image: Image
    random_cancer_type: str

    def __init__(self):
        if 'image_loader.obj' in os.listdir(os.getcwd()):
            with open('image_loader.obj', 'rb') as f:
                self.image_df = pickle.load(f)

        else:
            self.image_df = ImageLoader('./Images').df

        self.image_df['cancer_type'] = self.image_df['cancer_type']
        random.seed(st.session_state['SEED_VAL'])

    def random_sample(self):
        random_idx = random.randint(0, len(self.image_df))
        random_image = self.image_df['img_array'].iloc[random_idx]
        self.random_cancer_type = self.image_df['cancer_type'].iloc[random_idx]
        return random_image, self.random_cancer_type

    def capture_transformed_image(self, image):
        self.transformed_image = image

    def create_PIL_image(self, image):
        self.PIL_image = Image.fromarray(image)


class AlbTrxs:

    def __init__(self, horizontal_flip: bool, vertical_flip: bool, color_jitter: bool, rotate: bool, rgb_shift: bool):
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.color_jitter = color_jitter
        self.rotate = rotate
        self.rgb_shift = rgb_shift

    def selected_transforms(self, image):
        return A.Compose([A.HorizontalFlip(p=self.horizontal_flip),
                          A.VerticalFlip(p=self.vertical_flip),
                          A.ColorJitter(p=self.color_jitter),
                          A.Rotate(limit=10, interpolation=cv2.BORDER_CONSTANT, p=self.rotate),
                          A.RGBShift(p=self.rgb_shift)
                          ])(image=image)['image']


class TrainedModel:
    transformed_image: Any
    resize_factor: int = 347  # each image is 1388x1040 pixels, so this value represents a 4x downsize
    tl_means: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    tl_stds: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __init__(self, image_data: ImageData):
        self.image_data = image_data

        pytorch_algos = PytorchAlgos()
        self.model = pytorch_algos.RESNET101.model
        self.model.load_state_dict(torch.load('./best_model.pth'))

    def default_transforms(self):
        self.transformed_image = A.Compose([A.Normalize(mean=self.tl_means, std=self.tl_stds),
                                            A.LongestMaxSize(self.resize_factor),
                                            ToTensorV2()
                                            ])(image=self.image_data.transformed_image)['image']

    def make_prediction(self):
        data = {'cancer_type': [self.image_data.random_cancer_type], 'img_array': [self.transformed_image]}
        input_df = pd.DataFrame(data=data)

        dataset = PytorchImagesDataset(input_df)
        dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(TORCH_DEVICE), labels.to(TORCH_DEVICE)

            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

            decoder = label_decoder()
            prediction = int(preds)
            prediction = decoder[prediction]
            self.image_data.random_cancer_type = decoder[self.image_data.random_cancer_type]

        return self.image_data.random_cancer_type, prediction


header = st.beta_container()
image = st.beta_container()
run_model = st.beta_container()


with header:
    st.title('Lymphoma Subtype Classifier')
    st.write("""
    This app uses a trained model to classify sample Non-Hodgkins Lymphoma biopsy images into three sub-types: \
    Chronic Lymphocytic Leukemia (CLL), Follicular Lymphoma (FL), or Mantle Cell Lymphoma (MCL).  Toggle the \
    "Transformations" buttons to alter the image as you like before clicking the "Run Model" button, which will run the \
    model on the sample image.  The predicted and actual type will appear alongside.  See below the cross-entropy loss \
    graph for an understanding of what the model was "thinking", as well as some of the model performance statistics.
    """)
    st.write('')


with image:

    # define layout and add interactive elements
    image_col, selection_col = st.beta_columns([5, 2])

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
    alb_trxs = AlbTrxs(*session_states)
    image_data = ImageData()
    image, cancer_type = image_data.random_sample()
    image = alb_trxs.selected_transforms(image)
    image_data.capture_transformed_image(image)
    image_data.create_PIL_image(image)
    image_col.image(image_data.PIL_image, use_column_width=True)


with run_model:
    trained_model = TrainedModel(image_data)
    trained_model.default_transforms()

    run_model_col, prediction_col, actual_col = st.beta_columns(3)
    prediction_col.subheader('Predicted Type:')
    actual_col.subheader('Actual Type:')

    run_model_col.subheader('')
    run_model_button = run_model_col.button('Run Model')
    if run_model_button:
        actual, prediction = trained_model.make_prediction()
        prediction_col.write(prediction)
        actual_col.write(actual)
    else:
        prediction_col.write('Waiting')
        actual_col.write('Waiting')


