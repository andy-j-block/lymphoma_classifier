from typing import Tuple
from torch.utils.data import Dataset, DataLoader
import torch
import albumentations as A


class PytorchImagesDataset(Dataset):
    def __init__(self, cancer_type, img_array, transform=None):
        self.cancer_type = cancer_type
        self.img_array = img_array
        self.transform = transform

    def __len__(self):
        return len(self.cancer_type)

    def __getitem__(self, idx):
        img = self.img_array.iloc[idx]
        label = torch.tensor(int(self.cancer_type.iloc[idx]))

        if self.transform:
            img = self.transform(img)
        return img, label


###TODO edit this class name
class AlbumentationsTransformations:
    p: float
    resize_factor: float
    tl_means: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    tl_stds: Tuple[float, float, float] = (0.229, 0.224, 0.225)


    def __init__(self, p=0.2, resize_factor):
        self.p = p
        self.resize_factor = resize_factor #TODO img_widths_int / 4

    def create_transformation(self):
        self.training_transformations = A.Compose([A.HorizontalFlip(p=p),
                                              A.VerticalFlip(p=p),
                                              A.ColorJitter(p=p),
                                              A.Rotate(limit=10, interpolation=cv2.BORDER_CONSTANT, p=p),
                                              A.RGBShift(p=p)
                                              ])

        self.normalize_and_resize =  A.Compose([A.Normalize(mean=tl_means, std=tl_stds),
                                                         A.LongestMaxSize(resize_factor)
                                                         ])


class TransformedData:
    n_augmentation_passes: int

    def __init__(self, transformations: AlbumentationsTransformations, n_augmentation_passes):
        self.transformations = transformations
        self.n_augmentation_passes = n_augmentation_passes

    def transform_data(df_, phase, n_augs, transform_dict, overfit_test, **inputs):

        transformed_df = df_.copy()
        df_len = len(df_)
        cols = transformed_df.columns

        if not overfit_test:

            aug_pass = 0

            # generate new data n_augs number of times
            while aug_pass < n_augs:

                # apply tranforms to each entry in original dataframe
                for i in range(df_len):

                    # perform transformations
                    if phase == 'train':
                        transformed_array = transform_dict[phase](image=df_['img_array'].values[i])['image']
                        transformed_array = transform_dict['normalize_resize'](image=transformed_array)['image']

                        transformed_array = {cols[0]: df_[cols[0]].iloc[i],  # get original subtype label
                                             cols[1]: transformed_array
                                             }

                        transformed_df = transformed_df.append(transformed_array, ignore_index=True)

                aug_pass += 1

            # normalize and resize original data
            for i in range(df_len):
                transformed_array = transform_dict['normalize_resize'](image=transformed_df['img_array'].values[i])[
                    'image']
                transformed_df.at[i, 'img_array'] = transformed_array


        else:
            # overfit test
            rand_idx = random.randint(0, df_len)
            rand_entry = df_.iloc[rand_idx]
            transformed_data = transform_dict['normalize_resize'](image=rand_entry['img_array'])
            transformed_data = transformed_data['image']

            data = {cols[0]: [rand_entry[cols[0]]],
                    cols[1]: [transformed_data]}
            transformed_df = pd.DataFrame.from_dict(data=data)

        return transformed_df