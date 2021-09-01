from ImageLoader import ImageLoader
from KFolder import *
from PytorchAlgos import *
import torchvision.models as models


images = ImageLoader('./Images')
kfolder_overfit = KFoldIndices(image_data=images,
                               n_outer_splits=2,
                               n_inner_splits=2)
albumentation_transformations = AlbTrxs(resize_factor=4, n_passes=0)

df_train_overfit = DFTrainKFolded(n_outer_fold=0,
                                  n_inner_fold=0,
                                  kfold_idxs=kfolder_overfit)
df_valid_overfit = DFValidKFolded(n_outer_fold=0,
                                  n_inner_fold=0,
                                  kfold_idxs=kfolder_overfit)
df_train_overfit_dataloader = OverfitDataloader(kfolded_data=df_train_overfit,
                                                transformations=albumentation_transformations)
df_valid_overfit_dataloader = OverfitDataloader(kfolded_data=df_valid_overfit,
                                                transformations=albumentation_transformations)


resnet18 = ResNetModel(models.resnet18(pretrained=True))
