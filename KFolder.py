from sklearn.model_selection import KFold
from ImageLoader import ImageLoader
from abc import ABC, abstractmethod
from typing import Tuple, Union, Dict, List, Any
from torch.utils.data import Dataset, DataLoader
import torch
import albumentations as A
import random
import pandas as pd
import cv2
# from copy import deepcopy


class PytorchImagesDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.img_array, self.cancer_type = self.df['img_array'], self.df['cancer_type']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.img_array.iloc[idx], torch.tensor(int(self.cancer_type.iloc[idx]))


class AlbTrxs:
    p: float
    resize_factor: int
    tl_means: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    tl_stds: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    n_passes: int

    def __init__(self, resize_factor: int, n_passes: int, p=0.2):
        self.p = p
        self.resize_factor = resize_factor ###TODO img_widths_int / 4
        self.n_passes = n_passes
        self.training_transformations = A.Compose([A.HorizontalFlip(p=self.p),
                                                   A.VerticalFlip(p=self.p),
                                                   A.ColorJitter(p=self.p),
                                                   A.Rotate(limit=10, interpolation=cv2.BORDER_CONSTANT, p=self.p),
                                                   A.RGBShift(p=self.p)
                                                   ])
        self.normalize_and_resize = A.Compose([A.Normalize(mean=self.tl_means, std=self.tl_stds),
                                               A.LongestMaxSize(self.resize_factor)
                                               ])


class KFoldIndices(KFold):
    """this class creates indices for the outer and inner folds of the nested k-fold.
       k-folded dataframes will be constructed using these indices"""
    outer_train_idxs: Dict[int, Any]
    test_idxs: Dict[int, Any]
    train_idxs: Dict[Tuple[int, int], Any]
    valid_idxs: Dict[Tuple[int, int], Any]

    def __init__(self, image_data: ImageLoader, n_outer_splits: int, n_inner_splits: int, shuffle: bool = True, random_state: int = 42):
        super().__init__(shuffle=shuffle, random_state=random_state)
        self.image_df = image_data.df
        self.n_outer_splits, self.n_inner_splits = n_outer_splits, n_inner_splits
        self.shuffle, self.random_state = shuffle, random_state
        self.nested_outer()
        self.nested_inner()

    def nested_outer(self):
        """outer pass, return dicts of train and test set indices"""
        self.n_splits = self.n_outer_splits
        self.outer_train_idxs = {}
        self.test_idxs = {}

        i = 0
        # get indices of train and test splits and store in dicts
        for outer_train_idxs, test_idxs in self.split(self.image_df):
            self.outer_train_idxs[i] = outer_train_idxs
            self.test_idxs[i] = test_idxs
            i += 1

    def nested_inner(self):
        """inner pass, return dicts of train and val set indices"""
        self.n_splits = self.n_inner_splits
        self.train_idxs = {}
        self.valid_idxs = {}

        # get indices of train and val splits and store in dicts
        i = 0
        for n_fold, outer_train_idxs in self.outer_train_idxs.items():
            j = 0
            for train_idxs, valid_idxs in self.split(outer_train_idxs):
                self.train_idxs[(i, j)] = outer_train_idxs[train_idxs]
                self.valid_idxs[(i, j)] = outer_train_idxs[valid_idxs]
                j += 1
            i += 1


##############


class KFoldedDatasets(ABC):
    """this abstract base class is used for constructing the train, valid, and test dataframes
       based on the k-fold indices generated by the KFoldIndices class object"""
    phase: str

    @abstractmethod
    def __init__(self, n_outer_fold: int, n_inner_fold: int, kfold_idxs: KFoldIndices):
        self.image_df = kfold_idxs.image_df
        self.kfold_idxs = kfold_idxs
        self.n_outer_fold, self.n_inner_fold = n_outer_fold, n_inner_fold
        self.nkf_df = self.get_nkf_dataframe(self.n_outer_fold, self.n_inner_fold)  ###TODO is this necessary?

    @abstractmethod
    def get_nkf_dataframe(self, n_outer_fold: int, n_inner_fold: int):
        pass


class DFTrainKFolded(KFoldedDatasets):
    phase: str = 'train'

    def __init__(self, n_outer_fold: int, n_inner_fold: int, kfold_idxs: KFoldIndices):
        super().__init__(n_outer_fold, n_inner_fold, kfold_idxs)

    def get_nkf_dataframe(self, n_outer_fold: int, n_inner_fold: int):
        # get the full dataframes from the train and valid idxs
        df = self.image_df.iloc[self.kfold_idxs.train_idxs[(n_outer_fold, n_inner_fold)]]
        return df.reset_index(drop=True)


class DFValidKFolded(KFoldedDatasets):
    phase: str = 'valid'

    def __init__(self, n_outer_fold: int, n_inner_fold: int, kfold_idxs: KFoldIndices):
        super().__init__(n_outer_fold, n_inner_fold, kfold_idxs)

    def get_nkf_dataframe(self, n_outer_fold: int, n_inner_fold: int):
        df = self.image_df.iloc[self.kfold_idxs.valid_idxs[(n_outer_fold, n_inner_fold)]]
        return df.reset_index(drop=True)


class DFTestKFolded(KFoldedDatasets):
    phase: str = 'test'

    def __init__(self, n_outer_fold: int, n_inner_fold: int, kfold_idxs: KFoldIndices):
        super().__init__(n_outer_fold, n_inner_fold, kfold_idxs)

    def get_nkf_dataframe(self, n_outer_fold: int, n_inner_fold: int = None):
        df = self.image_df.iloc[self.kfold_idxs.test_idxs[self.n_outer_fold]]
        return df.reset_index(drop=True)


class DataloaderBaseClass(ABC):
    """this abstract base class is used to create pytorch dataloaders based on the dataframes passed
       to it by any of the DFTrainKFolded, DFValidKFolded, or DFTestKFolded class objects"""
    dataset: PytorchImagesDataset
    dataloader: DataLoader

    @abstractmethod
    def __init__(self, kfolded_data: Union[DFTrainKFolded, DFValidKFolded, DFTestKFolded],
                 transformations: AlbTrxs, batch_size: int, num_workers: int = 2):
        self.nkf_df = kfolded_data.nkf_df  ###TODO determine if this line in necessary
        self.outer_fold, self.inner_fold = kfolded_data.n_outer_fold, kfolded_data.n_inner_fold  ##TODO is this useful as a reference
        self.transformations = transformations
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transformed_data = pd.DataFrame()

    def normalize_resize_data(self):
        """normalize and resize original data for PyTorch"""
        original_data = self.nkf_df.copy(deep=True)
        for i in range(len(original_data)):
            original_data.at[i, 'img_array'] = \
                self.transformations.normalize_and_resize(image=original_data['img_array'].values[i])['image']
        self.transformed_data = original_data

    def create_dataloader(self):
        self.dataset = PytorchImagesDataset(self.transformed_data)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers)


class DFTrainDataloader(DataloaderBaseClass):
    generated_data: Dict[str, List[pd.Series]]

    def __init__(self, kfolded_data: Union[DFTrainKFolded, DFValidKFolded, DFTestKFolded],
                 transformations: AlbTrxs, batch_size: int):
        super().__init__(kfolded_data, transformations, batch_size)
        self.generate_data()
        self.normalize_resize_data()
        self.combine_data()
        self.create_dataloader()

    def generate_data(self):
        original_data = self.nkf_df.copy(deep=True)
        label, image = original_data.columns[0], original_data.columns[1]

        self.generated_data = {label: [], image: []}
        aug_pass_counter = 0
        while aug_pass_counter < self.transformations.n_passes:
            for i in range(len(original_data)):
                transformed_array = self.transformations.training_transformations(image=original_data['img_array'].values[i])['image']
                transformed_array = self.transformations.normalize_and_resize(image=transformed_array)['image']

                labeled_array = {label: original_data[label].iloc[i],  # get original subtype label
                                 image: transformed_array
                                 }
                self.generated_data[label].append(labeled_array[label])
                self.generated_data[image].append(labeled_array[image])
            aug_pass_counter += 1

    def combine_data(self):
        """concat the generated data with the normalized and resized original data"""

        generated_data = pd.DataFrame(data=self.generated_data)
        self.transformed_data = pd.concat([generated_data, self.transformed_data], ignore_index=True)


class DFValidDataloader(DataloaderBaseClass):
    def __init__(self, kfolded_data: Union[DFTrainKFolded, DFValidKFolded, DFTestKFolded],
                 transformations: AlbTrxs, batch_size: int):
        super().__init__(kfolded_data, transformations, batch_size)
        self.normalize_resize_data()
        self.create_dataloader()


class DFTestDataloader(DataloaderBaseClass):
    def __init__(self, kfolded_data: Union[DFTrainKFolded, DFValidKFolded, DFTestKFolded],
                 transformations: AlbTrxs, batch_size: int):
        super().__init__(kfolded_data, transformations, batch_size)
        self.normalize_resize_data()
        self.create_dataloader()


### TODO finish this dataloader
class OverfitDataloader(DataloaderBaseClass):
    def __init__(self, kfolded_data: Union[DFTrainKFolded, DFValidKFolded, DFTestKFolded],
                 transformations: AlbTrxs, batch_size: int):
        super().__init__(kfolded_data, transformations, batch_size)
        self.normalize_resize_data()
        # self.create_dataloader()

    def normalize_resize_data(self):
        rand_idx = random.randint(0, len(self.nkf_df))
        rand_entry = self.nkf_df.iloc[[rand_idx]].copy(deep=True)
        rand_entry.at[rand_idx, 'img_array'] = self.transformations.normalize_and_resize(image=rand_entry.at[rand_idx, 'img_array'])['image']
        self.transformed_data = rand_entry

        # data = {cols[0]: [rand_entry[cols[0]]],
        #         cols[1]: [transformed_data]}
        # self.transformed_data = pd.DataFrame.from_dict(data=data)

##############

# DEBUG:

# kfold_outer = OuterKFolder(n_splits=5,
#                            shuffle=True,
#                            random_state=42)
#
# kfold_inner = InnerKFolder(n_splits=5,
#                            shuffle=True,
#                            random_state=42)


# print(kfold_inner.valid_idxs[0])
# print(kfold_inner.valid_idxs[1])
# print(set(kfold_inner.train_idxs[1]).intersection(set(kfold_inner.valid_idxs[1])))
# print(len(kfold_inner.train_idxs[0]))
# print(len(kfold_inner.valid_idxs[0]))
