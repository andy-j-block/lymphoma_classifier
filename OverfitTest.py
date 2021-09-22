from ModelTrainer import *
from ImageLoader import ImageLoader
import pickle


def main():
    """first we inhale the image data, either by creating the ImageLoader object or via pickled object"""
    if 'overfit_data.obj' in os.listdir(os.getcwd()):
        with open('overfit_data.obj', 'rb') as f:
            images = pickle.load(f)

    else:
        images = ImageLoader('./Images').overfit_df

    """instantiate the algorithms object, get the kfold indices, instantiate the transformations object"""
    pytorch_algos = PytorchAlgos()
    kfolder_overfit = KFoldIndices(image_data=images, n_outer_splits=2, n_inner_splits=2)
    alb_trxs = AlbTrxs()

    """perform training steps and output results"""
    model_trainer = ModelTrainer(pytorch_algos=pytorch_algos, kfold_idxs=kfolder_overfit, transformations=alb_trxs)
    model_trainer.tune_model()
    model_trainer.model_selection()
    print('Here is a summary of some of the overfit data results:')
    print(model_trainer.results)


if __name__ == '__main__':
    main()
