# Creates a CNN trained on the on the global dataset. Is the model already exists, does nothing :)
# WORK IN PROGRESS

import numpy as np
from tensorflow import nn, keras, data
from tensorflow.keras import layers
from MegaZord.utilitaries.utils import one4all_labeller, weighter, ImageFromDirectory


class One4All:

    def __init__(self, path):

        self.data_augmentation = keras.Sequential([
            layers.InputLayer((256, 256, 3)),
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomContrast((0, 1))])
        try:
            self.model = keras.models.load_model(path + "/zords/" + "one4all.pb")
            print("successful importation")

        except OSError:

            print("one4all need to be trained...")
            temp = one4all_labeller(path + "/data")

            tab = np.array(temp)
            self.labels = tab[:, 0]

            folders = tab[:, 1].astype("int32")
            class_weight = weighter(folders)
            import_ds = ImageFromDirectory(path + "/data", "one4all")
            print("letsgo")
            print(import_ds.x.shape)
            train_ds = data.Dataset.from_tensor_slices((import_ds.x, import_ds.y)).batch(32)
            print("As data is imbalanced, the following weights will be applied ",
                  list(class_weight.values()))

            print("Augmenting the train_ds")
            augmented_train_ds = train_ds.map(lambda x, y: (self.data_augmentation(x), y))
            augmented_train_ds = augmented_train_ds.prefetch(buffer_size=32).shuffle(30)  # facilitates training
            print("Augmentation is done. Importation of InceptionV3 beginning...")
            print(self.labels)
            print(np.unique(import_ds.y))
            base_model = keras.applications.InceptionV3(
                include_top=False,
                weights="imagenet",
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                classifier_activation="softmax")

            # InceptionV3 layer are frozen so that we only need to train the last layer

            base_model.trainable = False

            print("Compilation of the CNN")
            input_shape = (256, 256, 3)
            inputs = keras.Input(shape=input_shape)
            x = keras.layers.experimental.preprocessing.Rescaling(1.0 / 255.)(inputs)
            x = base_model(x, training=False)
            x = keras.layers.GlobalAveragePooling2D()(x)

            outputs = keras.layers.Dense(len(folders), activation=nn.softmax)(x)

            self.model = keras.Model(inputs, outputs)
            print("compiling")
            self.model.compile(optimizer='adam',
                               loss='sparse_categorical_crossentropy',
                               metrics=['accuracy'])
            epochs = 4

            self.model.fit(augmented_train_ds, epochs=epochs, class_weight=class_weight)

            self.model.save(path + "/zords/" + "one4all" + ".pb")

            print("model saved")


if __name__ == "__main__":
    directory = "/Users/lucas/swiss_knife"
    one4all_init = One4All(directory)
    model = one4all_init.model
