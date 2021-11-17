import albumentations as albu
from tensorflow import keras
import numpy as np

def get_rand_bbox(width, height, l):
    r_x = np.random.randint(width)
    r_y = np.random.randint(height)
    r_l = np.sqrt(1 - l)
    r_w = np.int(width * r_l)
    r_h = np.int(height * r_l)
    return r_x, r_y, r_l, r_w, r_h

class DataGenerator(keras.utils.Sequence):
    def __init__(self,
        images,
        labels=None,
        mode='fit',
        batch_size=8,
        dim=(32, 32),
        channels=3,
        n_classes=100,
        shuffle=True,
        augment=False,
        preprocess_func=None,
        batch_preprocess_func=None,
        augment_args=None):
        
        #initializing the configuration of the generator
        self.images = images
        self.labels = labels
        self.mode = mode
        self.batch_size = batch_size
        self.dim = dim
        self.channels = channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment
        #self.rand_aug = iaa.RandAugment(n=3, m=7)
        self.preprocess_func = preprocess_func
        self.batch_preprocess_func = batch_preprocess_func

        if augment_args is not None:
            self.mix_up_alpha = augment_args["mix_up_alpha"]
            self.cutmix_alpha = augment_args["cutmix_alpha"]
            self.mix_up_switch_prob = augment_args["mix_up_switch_prob"]
            cutout_kwargs = augment_args["cutout_args"]
        else:
            self.mix_up_alpha = 0.8
            self.cutmix_alpha = 1.0
            self.mix_up_switch_prob = 0.5
            cutout_kwargs = {
                "p":0.25,
                "s_l":0.02,
                "s_h":1/3,
                "r_1":0.3,
                "r_2":1/0.3,
                "v_l":0,
                "v_h":255,
                "pixel_level":True
            }
        self.random_eraser = get_random_eraser(
            **cutout_kwargs
        )
        self.on_epoch_end()

    #method to be called after every epoch
    def on_epoch_end(self):
        self.indexes = np.arange(self.images.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

        if self.mix_up_alpha > 0.0:
            self.indexes_m = np.arange(self.images.shape[0])
            np.random.shuffle(self.indexes_m)

        if self.cutmix_alpha > 0.0:
            self.indexes_c = np.arange(self.images.shape[0])
            np.random.shuffle(self.indexes_c)
    
    #return numbers of steps in an epoch using samples and batch size
    def __len__(self):
        return int(np.floor(len(self.images) / self.batch_size))
    
    #this method is called with the batch number as an argument to obtain a given batch of data
    def __getitem__(self, index):
        #generate one batch of data
        #generate indexes of batch
        batch_indexes = self.indexes[index * self.batch_size:(index+1) * self.batch_size]

        #generate mini-batch of X
        X = np.empty((self.batch_size, *self.dim, self.channels))

        for i, ID in enumerate(batch_indexes):
            #generate pre-processed image
            img = self.images[ID]

            if self.preprocess_func is not None:
                img = self.preprocess_func(img, self.dim)

            #resizing as per new dimensions
            X[i] = img
           
        #generate mini-batch of y
        if self.mode == 'fit':
            y = self.labels[batch_indexes]
            
            #augmentation on the training dataset
            if self.augment:
                X = self.__augment_batch(X)

            if self.batch_preprocess_func is not None:
                X = self.batch_preprocess_func(X)

            if self.augment:

                if self.mix_up_alpha > 0.0:
                    if self.cutmix_alpha > 0.0 and np.random.rand() < self.mix_up_switch_prob:
                        pass
                    else:
                        X, y = self.mix_up(X, y, index)

                if self.cutmix_alpha > 0.0:
                    X, y = self.cutmix(X, y, index)

            self.last = (X, y)
            return X, y
        
        elif self.mode == 'predict':

            if self.batch_preprocess_func is not None:
                X = self.batch_preprocess_func(X)

            return X
        
        else:
            raise AttributeError("The mode should be set to either 'fit' or 'predict'.")
            
    #augmentation for one image
    def __random_transform(self, img):
        composition = albu.Compose([
                                   #albu.VerticalFlip(p=0.5),
                                   #albu.GridDistortion(p=0.2),
                                   #albu.RandomResizedCrop(32, 32, scale=(0.08, 1.12)),
                                    albu.PadIfNeeded(min_height=self.dim[0]+4, min_width=self.dim[1]+4),
                                    albu.RandomCrop(*self.dim),
                                    albu.HorizontalFlip(p=0.5)])
                                   #albu.ElasticTransform(p=0.2)])
        return composition(image=img)['image']
    
    #augmentation for batch of images
    def __augment_batch(self, img_batch):
        for i in range(img_batch.shape[0]):
            img_batch[i] = self.__random_transform(img_batch[i])
            #img_batch[i] = self.rand_aug(img_batch[i])
        return img_batch

    def prepare_aug_data(self, index, method):
        if method == "mixup":
            indexes = self.indexes_m
        elif method == "cutmix":
            indexes = self.indexes_c
        else:
            raise NotImplementedError("`method` is wrong.")

        batch_indexes_2 = indexes[index * self.batch_size:(index+1) * self.batch_size]
        X_2 = np.empty((self.batch_size, *self.dim, self.channels))

        for i, _ in enumerate(batch_indexes_2):
            img = self.images[batch_indexes_2[i]]
            if self.preprocess_func is not None:
                img = self.preprocess_func(img, self.dim)
            X_2[i] = img

        X_2 = self.__augment_batch(X_2)
        if self.batch_preprocess_func is not None:
            X_2 = self.batch_preprocess_func(X_2)
        y_2 = self.labels[batch_indexes_2]
        return X_2, y_2

    def mix_up(self, X1, y1, index):
        X2, y2 = self.prepare_aug_data(index, method="mixup")
        return self._mix_up(X1, y1, X2, y2)

    def _mix_up(self, X1, y1, X2, y2):
        assert X1.shape[0] == y1.shape[0] == X2.shape[0] == y2.shape[0]
        batch_size = X1.shape[0]
        l = np.random.beta(self.mix_up_alpha, self.mix_up_alpha, batch_size)
        X_l = l.reshape(batch_size, 1, 1, 1)
        y_l = l.reshape(batch_size, 1)
        X = X1 * X_l + X2 * (1-X_l)
        y = y1 * y_l + y2 * (1 - y_l)
        return X, y

    def cutmix(self, X1, y1, index):
        X2, y2 = self.prepare_aug_data(index, method="cutmix")
        return self._cutmix(X1, y1, X2, y2)

    def _cutmix(self, X1, y1, X2, y2):
        assert X1.shape[0] == y1.shape[0] == X2.shape[0] == y2.shape[0]
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        width = X1.shape[1]
        height = X1.shape[0]
        r_x, r_y, r_l, r_w, r_h = get_rand_bbox(width, height, lam)
        bx1 = np.clip(r_x - r_w // 2, 0, width)
        by1 = np.clip(r_y - r_h // 2, 0, height)
        bx2 = np.clip(r_x + r_w // 2, 0, width)
        by2 = np.clip(r_y + r_h // 2, 0, height)
        X1[:, bx1:bx2, by1:by2, :] = X2[:, bx1:bx2, by1:by2, :]
        X = X1
        y = y1 * lam + y2 * (1 - lam)
        return X, y

def get_random_eraser(p=0.5, s_l=0.02, s_h=1/3, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        if input_img.ndim == 3:
            img_h, img_w, img_c = input_img.shape
        elif input_img.ndim == 2:
            img_h, img_w = input_img.shape

        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            if input_img.ndim == 3:
                c = np.random.uniform(v_l, v_h, (h, w, img_c))
            if input_img.ndim == 2:
                c = np.random.uniform(v_l, v_h, (h, w))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w] = c

        return input_img

    return eraser
