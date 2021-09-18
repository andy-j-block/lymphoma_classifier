from ModelTrainer import *
import pickle


def main():
    """first we inhale the image data, either by creating the ImageLoader object or via pickled object"""
    # images = ImageLoader('./Images').df
    with open('overfit_data.obj', 'rb') as f:
        images = pickle.load(f)

    """next we create the Pytorch algos object so we know how many outer splits for the k-folder object"""
    pytorch_algos = PytorchAlgos()

    kfolder_overfit = KFoldIndices(image_data=images, n_outer_splits=2, n_inner_splits=2)
    alb_trxs = AlbTrxs()

    model_trainer = ModelTrainer(pytorch_algos=pytorch_algos, kfold_idxs=kfolder_overfit, transformations=alb_trxs)
    model_trainer.tune_model()
    model_trainer.model_selection()
    print(model_trainer.results)


if __name__ == '__main__':
    main()
