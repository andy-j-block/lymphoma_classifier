import os

class hardware_check:

    def running_hosted(self):
        self.hosted = True if 'content' in os.getcwd() else False
        print(f'This notebook is running hosted') if self.hosted else print(f'This notebook is running locally')

    def colab_setup(self):
        # check if git clone already performed
        if not os.path.isdir('./lymphoma_classifier'):
            !git
            clone
            https: // github.com / andy - j - block / lymphoma_classifier.git

        current_dir = os.getcwd()
        print(f"Current directory is '{current_dir}'")