import pickle
from ImageLoader import ImageLoader
from ExploratoryDataAnalysis import ExploratoryDataAnalysis
from KFolder import *
from Hyperparameters import Hyperparameters
from PytorchModel import *


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

kfolder = KFoldIndices(image_data=images,
                       n_outer_splits=3,
                       n_inner_splits=8)
albumentation_transformations = AlbumentationsTransformations(resize_factor=4, n_passes=3)
hyperparameters = Hyperparameters(batch_size=4, n_workers=2)
#
# df_train_kfolded = DFTrainKFolded(n_outer_fold=0,
#                                   n_inner_fold=0,
#                                   kfold_idxs=kfolder
#                                   )
#
# df_train_dataloader = DFTrainDataloader(kfolded_data=df_train_kfolded,
#                                         transformations=albumentation_transformations,
#                                         hyperparameters=hyperparameters
#                                         )

df_train_overfit = DFTrainKFolded(n_outer_fold=0,
                                  n_inner_fold=0,
                                  kfold_idxs=kfolder)
df_valid_overfit = DFValidKFolded(n_outer_fold=0,
                                  n_inner_fold=0,
                                  kfold_idxs=kfolder)
df_train_overfit_dataloader = OverfitDataloader(kfolded_data=df_train_overfit,
                                                transformations=albumentation_transformations,
                                                hyperparameters=hyperparameters)
df_valid_overfit_dataloader = OverfitDataloader(kfolded_data=df_valid_overfit,
                                                transformations=albumentation_transformations,
                                                hyperparameters=hyperparameters)


resnet18 = ResnetModel(models.resnet18(pretrained=True))


