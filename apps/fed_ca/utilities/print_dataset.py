from matplotlib import pyplot as plt
import numpy as np


class Print_data:

    def __init__(self, client_data):
        self.client_data = client_data

    def print_image_dataset(self):
        # print one image from the dataset
        for client_id, data in self.client_data.items():
            counter = 0
            data = data.as_numpy()
            for img in data.x:
                counter = counter + 1
                plt.imshow(np.reshape(img, (28, 28)))
                plt.show()
                if counter == 4:
                    break

    def print_dataset(self):
        # print 10 images from the dataset
        row = 0
        f, axarr = plt.subplots(10, 5, constrained_layout=False)
        for client_id, data in self.client_data.items():
            column = 0
            data = data.as_numpy()
            for img in data.x:
                axarr[row, column].imshow(np.reshape(img, (28, 28)))
                column = column + 1
                if column == 5:
                    break
            row = row + 1
        plt.show()
