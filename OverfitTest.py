from ImageLoader import ImageLoader
from KFolder import *
from PytorchAlgos import *
import torchvision.models as models
import pickle

### TODO ton of work on this once training loop creation done
# images = ImageLoader('./Images')
with open('image_loader.obj', 'rb') as f:
    images = pickle.load(f)
kfolder_overfit = KFoldIndices(image_data=images,
                               n_outer_splits=2,
                               n_inner_splits=2)
albumentation_transformations = AlbTrxs()

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


resnet18 = CustomResNet(models.resnet18(pretrained=True))
