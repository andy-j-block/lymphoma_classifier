from ImageLoader import ImageLoader
from KFolder import *
from Hyperparameters import Hyperparameters
from PytorchModel import *

images = ImageLoader('./Images')
kfolder_overfit = KFoldIndices(image_data=images,
                               n_outer_splits=2,
                               n_inner_splits=2)
albumentation_transformations = AlbumentationsTransformations(resize_factor=4, n_passes=0)
hyperparameters = Hyperparameters(batch_size=1, n_workers=2)

df_train_overfit = DFTrainKFolded(n_outer_fold=0,
                                  n_inner_fold=0,
                                  kfold_idxs=kfolder_overfit)
df_valid_overfit = DFValidKFolded(n_outer_fold=0,
                                  n_inner_fold=0,
                                  kfold_idxs=kfolder_overfit)
df_train_overfit_dataloader = OverfitDataloader(kfolded_data=df_train_overfit,
                                                transformations=albumentation_transformations,
                                                hyperparameters=hyperparameters)
df_valid_overfit_dataloader = OverfitDataloader(kfolded_data=df_valid_overfit,
                                                transformations=albumentation_transformations,
                                                hyperparameters=hyperparameters)


resnet18 = ResnetModel(models.resnet18(pretrained=True))