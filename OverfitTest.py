from ImageLoader import *
from KFolder import *
from PytorchAlgos import *
from ModelTrainer import *
import torchvision.models as models
import pickle


def main():
    ### TODO ton of work on this once training loop creation done
    """first we inhale the image data, either by creating the ImageLoader object or via pickled object"""
    # images = ImageLoader('./Images').df
    with open('overfit_data.obj', 'rb') as f:
        images = pickle.load(f)

    """next we create the Pytorch algos object so we know how many outer splits for the k-folder object"""
    pytorch_algos = PytorchAlgos()

    kfolder_overfit = KFoldIndices(image_data=images, n_outer_splits=2, n_inner_splits=2)
    alb_trxs = AlbTrxs()

    model_trainer = ModelTrainer(pytorch_algos=pytorch_algos, kfold_idxs=kfolder_overfit, transformations=alb_trxs)
    model_trainer.model_train()

    #
    # df_train_overfit = DFTrainKFolded(n_outer_fold=0,
    #                                   n_inner_fold=0,
    #                                   kfold_idxs=kfolder_overfit)
    # df_valid_overfit = DFValidKFolded(n_outer_fold=0,
    #                                   n_inner_fold=0,
    #                                   kfold_idxs=kfolder_overfit)
    # df_train_overfit_dataloader = OverfitDataloader(kfolded_data=df_train_overfit,
    #                                                 transformations=albumentation_transformations)
    # df_valid_overfit_dataloader = OverfitDataloader(kfolded_data=df_valid_overfit,
    #                                                 transformations=albumentation_transformations)
    #
    #
    # resnet18 = CustomResNet(models.resnet18(pretrained=True))


if __name__ == '__main__':
    main()
