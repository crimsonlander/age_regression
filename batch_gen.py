from __future__ import division, print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


class Augmentor(object):
    def __init__(self, rotation_range=30, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                 zoom_range=0.2, horizontal_flip=True, fill_mode='reflect'):

        self.image_augmentor = ImageDataGenerator(
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            horizontal_flip=horizontal_flip,
            fill_mode=fill_mode)

    def __call__(self, image_batch, mask_batch=None):
        state = np.random.get_state()
        aug_image_batch = np.array(image_batch)
        for i in range(image_batch.shape[0]):
            aug_image_batch[i] = self.image_augmentor.random_transform(image_batch[i])

        if mask_batch is not None:
            np.random.set_state(state)
            aug_mask_batch = np.array(mask_batch)
            for i in range(mask_batch.shape[0]):
                aug_mask_batch[i] = self.image_augmentor.random_transform(mask_batch[i])

            return aug_image_batch, aug_mask_batch
        else:
            return aug_image_batch


class AgeGenderBatchGenerator(object):
    """
    Generates batches of prepared images labeled with age group and gender.
    """
    def __init__(self, X, age, gender, batch_size, patch_shape, stride, random_offset, augment=True):
        """
        :param X: python list or numpy array of images.
        Each image should be a numpy array of shape (height, width, channels)
        :param age: python list or numpy array of age group or age labels
        Each label should be a numpy array of shape (1,)
        :param gender: python list or numpy array of gender labels
        Each label should be a numpy array of shape (1,)
        :param batch_size: number of samples returned by single call.
        :param patch_shape: tuple, shape of patches sampled from the images (height, width)
        :param stride: tuple, stride step sizes (vertical, horizontal)
        :param random_offset: tuple, maximum values of random initial offsets (vertical, horizontal)
        """

        self.X = X
        self.age = age
        self.gender = gender

        order = np.random.permutation(self.X.shape[0])
        self.X = self.X[order]
        self.age = self.age[order]
        self.gender = self.gender[order]

        self.batch_size = batch_size

        if augment:
            self.augmentor = Augmentor(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1,
                                   shear_range=0.05, zoom_range=0.3, fill_mode='reflect', )
        else:
            self.augmentor = None

        self.index = 0
        self.offset = np.random.randint(random_offset[1]) if random_offset[1] else 0
        self.i = np.random.randint(random_offset[0]) if random_offset[0] else 0
        self.j = self.offset

        self.num_images = len(X)

        self.patch_shape = patch_shape

        self.stride = stride
        self.random_offset = random_offset

        self.image = self.X[self.index]
        self.cur_age = self.age[self.index]
        self.cur_gender = self.gender[self.index]

    def get_supervised_batch(self):
        """
        :return: A tuple of numpy arrays: (data_batch, age_labels, gender_labels)
        """
        batch_patches = []
        batch_age = []
        batch_gender = []

        for i in range(self.batch_size):

            patch = self.image[self.i:self.i + self.patch_shape[0], self.j:self.j + self.patch_shape[1]]
            batch_patches.append(np.expand_dims(patch, 0))
            batch_age.append(self.cur_age)
            batch_gender.append(self.cur_gender)

            self.j += self.stride[1]

            if self.j >= self.image.shape[1] - self.patch_shape[1]:

                self.i += self.stride[0]
                self.j = self.offset

                if self.i >= self.image.shape[0] - self.patch_shape[0]:

                    self.i = np.random.randint(self.random_offset[0]) if self.random_offset[0] else 0
                    self.offset = np.random.randint(self.random_offset[1]) if self.random_offset[1] else 0
                    self.j = self.offset

                    self.index += 1

                    if self.index >= self.num_images:
                        self.index = 0
                        order = np.random.permutation(self.X.shape[0])
                        self.X = self.X[order]
                        self.age = self.age[order]
                        self.gender = self.gender[order]

                    self.image = self.X[self.index]
                    self.cur_age = self.age[self.index]
                    self.cur_gender = self.gender[self.index]

        if self.augmentor:
            return self.augmentor(np.concatenate(batch_patches)), \
                   np.concatenate(batch_age), np.concatenate(batch_gender)
        else:
            return np.concatenate(batch_patches), \
                   np.concatenate(batch_age), np.concatenate(batch_gender)

    def get_unsupervised_batch(self):
        """
        :return: A batch of data as a numpy array
        """
        batch_X, _, _ = self.get_supervised_batch()

        return batch_X
