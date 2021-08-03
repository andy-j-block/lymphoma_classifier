from sklearn.model_selection import KFold

# def default_arg_handler(arg, default_value):
#     if arg is not None:
#         return arg
#     else:
#         return default_value



class OuterKFolder(KFold):
    def nested_outer(self, df):
        """outer pass, return dicts of train and test set indices"""

        self.train_idxs = {}
        self.test_idxs = {}

        counter = 0

        # get indices of train and test splits and store in dicts
        for train_idxs, test_idxs in self.split(df):
            self.train_idxs[counter] = train_idxs
            self.test_idxs[counter] = test_idxs

            counter += 1


class InnerKFolder(KFold):

    def nested_inner(self, outer_kfold: OuterKFolder):
        """inner pass, return dicts of train and val set indices"""

        self.train_idxs = {}
        self.valid_idxs = {}

        counter = 0

        # get indices of train and val splits and store in dicts
        for i, _ in enumerate(outer_kfold.train_idxs):
            for train_idxs, valid_idxs in self.split(outer_kfold.train_idxs[i]):
                self.train_idxs[counter] = outer_kfold.train_idxs[i][train_idxs]
                self.valid_idxs[counter] = outer_kfold.train_idxs[i][valid_idxs]

                counter += 1


##############


def nested_kfold(df, n_splits_outer, n_splits_inner, random_state, shuffle_outer=True, shuffle_inner=True):
    # define indices for nested k-fold splits
    train_idxs_outer, test_idxs_outer = nested_outer(df, n_splits_outer, shuffle_outer, random_state)
    train_idxs_inner, valid_idxs_inner = nested_inner(train_idxs_outer, n_splits_inner, shuffle_inner, random_state)

    nkf = {'train_outer': train_idxs_outer,
           'test_outer': test_idxs_outer,
           'train_inner': train_idxs_inner,
           'valid_inner': valid_idxs_inner
           }

    return nkf


##############


def nkf_full_dataframes(df, nkf, n_fold, **inputs):
    # get the full dataframes from the train and valid idxs
    df_train = df.iloc[nkf['train_inner'][n_fold]]
    df_valid = df.iloc[nkf['valid_inner'][n_fold]]

    # reset idxs for albumentations library
    df_train.reset_index(drop=True, inplace=True)
    df_valid.reset_index(drop=True, inplace=True)

    # return df's of a single inner fold
    return df_train, df_valid



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