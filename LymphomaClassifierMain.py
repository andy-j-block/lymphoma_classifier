from ModelTrainer import *
import pickle


def main():

    with open('image_loader.obj', 'rb') as f:
        image_data = pickle.load(f)

    pytorch_algos = PytorchAlgos()

    kfolder = KFoldIndices(image_data=image_data, n_outer_splits=pytorch_algos.n_algos, n_inner_splits=3)
    alb_trxs = AlbTrxs()

    model_trainer = ModelTrainer(pytorch_algos=pytorch_algos, kfold_idxs=kfolder, transformations=alb_trxs)
    model_trainer.tune_model()
    model_trainer.save_results('tuned_results.csv')
    model_trainer.model_selection()
    model_trainer.save_results('selected_results.csv')
    model_trainer.model_scoring()
    model_trainer.save_results('scored_results.csv')
    model_trainer.get_predictions()
    model_trainer.save_results('final_results.csv')


if __name__ == '__main__':
    main()



