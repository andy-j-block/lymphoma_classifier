from abc import ABC, abstractmethod
from KFolder import KFoldedDatasets
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
import torch
import albumentations as A
import random
import pandas as pd
import cv2


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


    def __init__(self, resize_factor, p=0.2):
        self.p = p
        self.resize_factor = resize_factor #TODO img_widths_int / 4

    def get_transformations(self):
        self.training_transformations = A.Compose([A.HorizontalFlip(p=self.p),
                                              A.VerticalFlip(p=self.p),
                                              A.ColorJitter(p=self.p),
                                              A.Rotate(limit=10, interpolation=cv2.BORDER_CONSTANT, p=self.p),
                                              A.RGBShift(p=self.p)
                                              ])

        self.normalize_and_resize =  A.Compose([A.Normalize(mean=self.tl_means, std=self.tl_stds),
                                                         A.LongestMaxSize(self.resize_factor)
                                                         ])


class TransformedData:
    n_augmentation_passes: int

    def __init__(self, kfolded_data: KFoldedDatasets, transformations: AlbumentationsTransformations, n_augmentation_passes: int, overfit_test=False):
        self.kfolded_data = kfolded_data ###TODO determine if this line in necessary
        self.transformations = transformations
        self.n_augmentation_passes = n_augmentation_passes
        self.overfit_test = overfit_test

    def transform_data(self, phase, transform_dict):
        data = self.image_data.df.deepcopy()
        data_len = len(data)
        cols = data.columns

        if not self.overfit_test:
            aug_pass_counter = 0
            # generate new data n_augs number of times
            while aug_pass_counter < self.n_augmentation_passes:

                # apply tranforms to each entry in original dataframe
                for i in range(data_len):

                    # perform transformations
                    if phase == 'train':
                        transformed_array = transform_dict[phase](image=data['img_array'].values[i])['image']
                        transformed_array = transform_dict['normalize_resize'](image=transformed_array)['image']

                        transformed_array = {cols[0]: data[cols[0]].iloc[i],  # get original subtype label
                                             cols[1]: transformed_array
                                             }

                        transformed_df = data.append(transformed_array, ignore_index=True)

                aug_pass_counter += 1

            # normalize and resize original data
            for i in range(data_len):
                transformed_array = transform_dict['normalize_resize'](image=transformed_df['img_array'].values[i])[
                    'image']
                transformed_df.at[i, 'img_array'] = transformed_array


        else:
            # overfit test
            rand_idx = random.randint(0, data_len)
            rand_entry = data.iloc[rand_idx]
            transformed_data = transform_dict['normalize_resize'](image=rand_entry['img_array'])
            transformed_data = transformed_data['image']

            data = {cols[0]: [rand_entry[cols[0]]],
                    cols[1]: [transformed_data]}
            transformed_df = pd.DataFrame.from_dict(data=data)

        return transformed_df

    def create_dataloaders(self, batch_size, n_workers, get_dataset=False, df_train=None, df_valid=None, df_test=None,
                           **inputs):

        # This function takes our input data and creates dataloader dictionaries.
        # It is customizable to be used on train, valid, and test phases

        keys = []
        data_values = []
        loader_values = []

        ###

        def populate_lists(df, phase, get_dataset=get_dataset):
            data = lymphoma_images_dataset(df['cancer_type'], df['img_array'])
            loader = DataLoader(data, batch_size, n_workers)
            keys.append(phase)
            if get_dataset is True:
                data_values.append(data)
            loader_values.append(loader)

        ###

        if df_train is not None:
            populate_lists(df_train, 'train')

        if df_valid is not None:
            populate_lists(df_valid, 'valid')

        if df_test is not None:
            populate_lists(df_test, 'test')

        datasets = {key: value for (key, value) in zip(keys, data_values)}
        dataloaders = {key: value for (key, value) in zip(keys, loader_values)}

        if get_dataset is True:
            return datasets, dataloaders
        else:
            return dataloaders


    def inner_fold_dataloaders(df, nkf, n_fold, n_augs, transform_dict, transform, overfit_test, n_workers, batch_size,
                               get_dataset, **inputs):

        # pdb.set_trace()

        df_train, df_valid = nkf_full_dataframes(df, nkf, n_fold)

        # transform the data
        df_train = transform_data(df_train, 'train', n_augs, transform_dict, overfit_test, **inputs)
        df_valid = transform_data(df_valid, 'valid', 0, transform_dict, overfit_test, **inputs)

        dataloaders = create_dataloaders(batch_size, n_workers, get_dataset, df_train, df_valid, **inputs)

        return dataloaders