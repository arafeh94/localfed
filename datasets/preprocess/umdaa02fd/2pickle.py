import csv
import pickle
import copy
import glob
import os
import pickle

import PIL
import numpy as np
import tensorflow as tf
from PIL import Image

from src.data.data_container import DataContainer
from src.data.data_provider import PickleDataProvider


def verify_image_number(imgs_nb, imgs_dst_nb):
    """
    used to check the number of images of user and compare it with the number recorded in the training dataset
    :return: bool
    """
    if imgs_nb == imgs_dst_nb:
        return "Number of Images matching"
    else:
        return f"Missing {imgs_nb - imgs_dst_nb} Images "


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def read_pickle_file(filepath):
    objects = []
    with (open(filepath, "rb")) as openfile:
        while True:
            try:
                objects.extend(pickle.load(openfile))
            except EOFError:
                break
    return objects


def get_user_images(filtered_clients, targeted_user):
    client_images = []
    for im in filtered_clients:
        if im[6] == targeted_user.split("\\", 1)[1]:
            client_images.append(im)

    return client_images


def get_data(filtered_clients, target_idx, train_ratio=1, input_shape=(128, 128), channels=3):
    """
    target_idx(int): ranged from 0~43, which are the users ids
    train_ratio(int): the ration of the data for the training set
    input_shape(tuple): the desired output of the images
    """

    # read all users images from nested folders of the umdaa-02-fd dataset
    root_path = '../../raw/umdaa02fd/Data/'
    users = glob.glob(os.path.join(root_path, '*'))
    target_user = copy.deepcopy(users[target_idx])

    train_target_imgs = []
    client_data = get_user_images(filtered_clients, target_user)
    for image in client_data:
        train_target_imgs.append(image[0])
    train_len = int(len(train_target_imgs) * train_ratio)
    print()

    @tf.function
    def parse_image(filename, y, pad=input_shape[0] // 8):
        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, input_shape)
        # augmentation
        image = tf.image.resize_with_crop_or_pad(image,
                                                 input_shape[0] + pad * 2,
                                                 input_shape[1] + pad * 2)
        image = tf.image.random_crop(image, input_shape + (3,))
        image = tf.image.random_flip_left_right(image)
        image = image * 2. - 1.  # norm to (-1, +1)
        return image, y

    # Train
    train_dset = tf.data.Dataset.from_tensor_slices(
        (train_target_imgs,
         np.ones(shape=[len(train_target_imgs), 1], dtype=np.float32)))

    train_dset = train_dset.map(
        parse_image,
        num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().shuffle(
        1000).batch(1, drop_remainder=False).prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    print(target_user)
    X_train = list(map(lambda x: x[0], train_dset))

    ys = [target_idx] * len(X_train)
    xs = []
    for user_images in X_train:
        for user_image in user_images:
            array = np.reshape(user_image.numpy(), input_shape[0] * input_shape[1] * channels)
            xs.append(array)

    print(f'Prosessing Data For User {target_idx} Is Finished!', verify_image_number(train_len, len(X_train)))

    return xs, ys


# read data from the csv file
clients_data = []
with open('../../raw/umdaa02fd/AnnotatedAA02_TL_BR.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        clients_data.append(row)

# original clients number before filtering
print(len(clients_data))

filtered_clients = []
# filter missing data, sessions and images without faces
# use the main_load_from_csv to read the images from the csv file, it contains the missing data infomation represented by 2, 3, 4, 5 on the latest column
# the columns that contains -1, -1, -1, -1 means that the image does not contain a face.
for x in clients_data:
    if x[1] != '-1' and x[5] == '1':
        filtered_clients.append(x)

# adding the user name in the list
for session in filtered_clients:
    session.append(session[0].split("/", 3)[2])

# umdaa-02-fd contains 44 users in totol
all_train_x = []
all_train_y = []
user_labels = 44
for user_label in range(user_labels):
    xs, ys = get_data(filtered_clients, user_label)
    all_train_x.append(xs)
    all_train_y.append(ys)

xs = []
ys = []
for image, label in zip(all_train_x, all_train_y):
    xs.extend(image)
    ys.extend(label)

all_data = {'data': xs, 'label': ys}

dc = DataContainer(xs, ys)
print("saving...")
PickleDataProvider.save(dc, '../../pickles/umdaa02_fd_filtered_test.pkl')
