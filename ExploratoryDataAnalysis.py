import ImageLoader
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
from scipy.special import binom
import os


class JupyterLauncher:
    cwd: str
    conda_exists: bool

    def __init__(self):
        self.get_cwd()
        self.create_conda_env()
        self.launch_notebook()

    def get_cwd(self):
        self.cwd = os.getcwd()

    def create_conda_env(self):
        for _, val in os.environ.items():
            if 'conda' in val.lower():
                self.conda_exists = True
                print('conda found on system')
                break
            else:
                print('conda not found on system, please install and rerun')

        if self.conda_exists:
            os.system('conda env create --file envname.yml')
            os.system('conda activate lymphoma_classifier')

    def launch_notebook(self):
        os.system(f'cd {self.cwd}')
        os.system('jupyter notebook ExploratoryDataAnalysis.ipynb')


class ExploratoryDataAnalysis:

    def __init__(self, image_data: ImageLoader):
        self.image_df = image_data.df

    def cancer_type_counts(self):
        """print the value counts for each of the cancer subtypes and display the values on a chart"""

        """print the value counts"""
        self.cancer_type_series = self.image_df['cancer_type']
        print(f'The cancer type value counts are:\n{self.cancer_type_series.value_counts()}')

        """create bar chart with cancer type as x, counts as y"""
        self.cancer_types = self.cancer_type_series.value_counts().index
        self.cancer_counts = self.cancer_type_series.value_counts().values
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        bar_chart = ax.bar(x=self.cancer_types, height=self.cancer_counts)

        """calculate cancer type percentages and print them atop each bar"""
        type_percents = [round(i/len(self.cancer_type_series), 2) for i in self.cancer_counts]
        for i, bar in enumerate(bar_chart):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    type_percents[i],
                    ha='center', va='bottom')

        plt.title('Count by cancer type with percentages as text')
        plt.show()


    def get_image_dims(self):
        """get the image-wise dimensions"""

        images = self.image_df['img_array']
        img_heights, img_widths = set(), set()

        for img in images:
            height, width = img.shape[0], img.shape[1]
            img_heights.add(height)
            img_widths.add(width)

        print('Image heights are: ', img_heights)
        print('Image widths are: ', img_widths)

        #
        # # widths and heights if input is torchvision dataset
        # else:
        #
        #     for i in range(len(data)):
        #         width, height = data[i][0].shape[2], data[i][0].shape[1]
        #         img_heights.append(height)
        #         img_widths.append(width)
        #
        #     # cast to set to eliminate duplicates
        #     img_heights = set(img_heights)
        #     img_widths = set(img_widths)
        #
        #     print('Image heights are: ', img_heights)
        #     print('Image widths are: ', img_widths)
        #
        # if not set_:
        #     img_heights = min(img_heights)
        #     img_widths = min(img_widths)
        #
        # return img_heights, img_widths


    def get_intensity_range(self):
        """get the pixel-wise intensity mins and maxs"""

        images = self.image_df['img_array']
        maxs = [np.amax(img) for img in images]
        mins = [np.amin(img) for img in images]

        print(f'The highest intesity in the range: {max(maxs)}')
        print(f'The lowest intesity in the range: {min(mins)}')

        # # intensities if input is torchvision dataset
        # else:
        #     for i in range(len(data)):
        #         maxs.append(torch.max(data[i][0]))
        #         mins.append(torch.min(data[i][0]))


    def get_random_image(self, figsize=(30, 10)):
        """get a random image of each of the cancer types"""

        rand_imgs = []

        for _, cancer_type in enumerate(self.cancer_types):
            index_by_type = self.image_df[self.image_df['cancer_type'] == cancer_type].index
            rand_idx = random.choice(index_by_type)
            rand_imgs.append(self.image_df.at[rand_idx, 'img_array'])

        fig, axs = plt.subplots(1, len(rand_imgs), figsize=figsize)
        for i, img in enumerate(rand_imgs):
            axs[i].set_title(f'Cancer type: {self.cancer_types[i]}')
            axs[i].imshow(img)
        plt.show()


    def plot_prob_transforms(self, p_values, n_poss_transforms, figsize=(30, 10)):
        """plot the probabilities that n transformations are applied to a given image based on the number of possible
            transformations and the probability p of the transformer"""

        prob_dict = {}
        num_p_values = len(p_values)

        """define probabilities of specific number of transforms"""
        for p in p_values:
            p_comp = 1-p
            prob_dict[p] = {i: p**i * p_comp**(n_poss_transforms-i) * binom(n_poss_transforms, i)
                            for i in range(n_poss_transforms+1)}

        """create subplots for however many p-values input"""
        fig, axs = plt.subplots(1, num_p_values, figsize=figsize)
        fig.suptitle('Probability of a given number of transforms with varying p values')
        sns.set_style('dark')

        """define dual axes dict and make dual x axes for pareto"""
        axes = {i: axs for (i, axs) in enumerate(axs)}
        for i, ax in enumerate(axs):
            axes[i+num_p_values] = ax.twinx()
            axes[i+num_p_values].set_ylim(0, 1.05)

        """first create bar plot for probability values and then pareto line plot on top"""
        for i, p in enumerate(p_values):
            keys = list(prob_dict[p].keys())
            values = [prob_dict[p][k] for k in keys]
            sns.barplot(x=keys, y=values, ax=axes[i])
            axes[i].set_title(f'p={p}')

            cumulative_values = np.cumsum(values)
            sns.lineplot(x=keys, y=cumulative_values, ax=axes[i+num_p_values], color='black',
                         marker='o', markersize=10, linewidth=3)
        plt.show()
