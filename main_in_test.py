import pickle
from ImageLoader import ImageLoader
from ExploratoryDataAnalysis import ExploratoryDataAnalysis
from KFolder import *
from PytorchAlgos import *

# images = image_loader()
# images.get_cancer_types('./Images')
# images.get_img_dirs()
# images.load_images()
# # images.data_preview()
# images.create_dataframe()

with open('image_loader.obj', 'rb') as f:
    images = pickle.load(f)

# EDA = exploratory_data_analysis(images)
# EDA.cancer_type_counts()
# EDA.get_image_dims()
# EDA.get_intensity_range()
# EDA.get_random_image()
# EDA.plot_prob_transforms(p_values=[0.2, 0.3, 0.4, 0.5], n_poss_transforms=5)


#overfit_image_data ###TODO create miniature dataset
pytorch_models = PytorchAlgos()
kfolder = KFoldIndices(image_data=images, n_outer_splits=pytorch_models.n_models, n_inner_splits=4)
albumentation_transformations = AlbTrxs()




