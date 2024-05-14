from snowflake.snowpark import Session
from utils import label_encoder, transform_image_bytes

class ImageLoader:
    
    # df: pd.DataFrame

    def __init__(self, session: Session):
        self.session = session
        
        self.df = self.session.table(f'all_images').to_pandas()
        self.df.rename(columns={"IMG_BYTES": "img_bytes", "ID": "id", "CANCER_TYPE": "cancer_type"}, inplace=True)
        self.df['cancer_type'] = self.df['cancer_type'].map(label_encoder())
        
    def load_images(self):
        self.df['img_array'] = self.df['img_bytes'].apply(transform_image_bytes)
        return self.df
    
def get_all_img_ids(session: Session, img_table: str):
    values = session.sql(f'select distinct id from {img_table};').collect()
    return [x['ID'] for x in values]

    # def __init__(self, top_img_dir: str):
    #     self.top_img_dir = top_img_dir
    #     """get the cancer types from the subdirectory names and their full paths"""
    #     self.cancer_types = [cancer_type for cancer_type in os.listdir(self.top_img_dir)]
    #     self.img_dirs = [os.path.join(self.top_img_dir, cancer_type) for cancer_type in self.cancer_types]

    #     """load the images with PIL/numpy and save them to a pandas dataframe"""
    #     self.imgs_and_labels = self.load_images()
    #     self.df = pd.DataFrame(self.imgs_and_labels, columns=['cancer_type', 'img_array'])

    #     """add label encoding because pytorch doesnt handle strings"""
    #     self.df['cancer_type'] = self.df['cancer_type'].map(label_encoder())
    #     self.overfit_df = self.df.iloc[random.sample(range(0, len(self.df)), 4)]



    # def load_images(self):
    #     """read images into a list"""
    #     imgs_and_labels = []

    #     for i, img_dir in enumerate(self.img_dirs):
    #         img_paths = os.listdir(img_dir)
    #         for j in img_paths:
    #             """pass thru all the image files per image directory, open the image with pillow, 
    #             convert to numpy array, add it to the images list with its cancer type label"""
    #             img_path = os.path.join(self.img_dirs[i], j)
    #             img_array = np.asarray(Image.open(img_path))
    #             imgs_and_labels.append((self.cancer_types[i], img_array))
    #     return imgs_and_labels

 

# def main(session: Session):
#     image_loader = ImageLoader(session)


# def main(top_img_dir: str = './Images', pickle_=True):

#     image_loader = ImageLoader(top_img_dir=top_img_dir)

#     if pickle_:
#         with open('image_loader.obj', 'wb') as f:
#             pickle.dump(image_loader.df, f)
#             print('ImageLoader object saved successfully')

#         with open('overfit_data.obj', 'wb') as f:
#             pickle.dump(image_loader.overfit_df, f)
#             print('Overfit test data object saved successfully')