import pickle
from ImageLoader import image_loader
from ExploratoryDataAnalysis import ExploratoryDataAnalysis

def main():

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


main()