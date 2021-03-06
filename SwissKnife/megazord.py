# The SwissKnife is the center class of our project. It centralizes many of the functions we need.

# ToDo LIST :
# Early dropout

import os
import logging
import coremltools as ct

from tensorflow.keras import layers
from tensorflow import nn, keras, data
from tensorflow.keras.models import load_model
from tensorflow import math, multiply, cast, float32, stack
from tqdm import tqdm
from MegaZord.utilitaries.utils import listdir_nohidden, data_repartition, weighter, flatten

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)


def get_data(zord: str, path: str, label_mode="int"):
    """

    :param zord: name of the zord. if main_zord stack same class labels together.
    :param path: main path
    :param label_mode:
    :return: a tensorflow.k0eras Dataset
    """
    from tensorflow import keras
    if zord == "main_zord":
        train_directory = path + "/train_set"
        val_directory = path + "/validation_set"
    else:
        train_directory = path + "/train_set/" + zord
        val_directory = path + "/validation_set/" + zord
    print("_____________ Training {} __________".format(zord))
    print(" Importing dataset...")

    if label_mode is None:
        try:
            train_ds = keras.preprocessing.image_dataset_from_directory(
                train_directory,
                label_mode=label_mode, shuffle=True, batch_size=256)
            val_ds = keras.preprocessing.image_dataset_from_directory(
                val_directory, label_mode=label_mode, shuffle=True, batch_size=256, validation_split=0.3,
                subset="validation", seed=123)

        except:
            train_ds = keras.preprocessing.image_dataset_from_directory(
                train_directory,
                label_mode=label_mode, shuffle=True, batch_size=256, subset = "training", validation_split = 0.2)
            val_ds = keras.preprocessing.image_dataset_from_directory(
                val_directory, label_mode=label_mode, shuffle=True, batch_size=256, validation_split=0.2,
                subset="validation", seed=123)

    else:
        """try:
            train_ds = keras.preprocessing.image_dataset_from_directory(
                train_directory, labels="inferred",
                label_mode="int", shuffle=True, batch_size=256)
            val_ds = keras.preprocessing.image_dataset_from_directory(
                val_directory, labels="inferred",
                label_mode="int", shuffle=True, batch_size=256, validation_split=0.3, subset="validation", seed=123)
        except:"""
        train_ds = keras.preprocessing.image_dataset_from_directory (
            train_directory,
            label_mode=label_mode, labels="inferred", shuffle=True, batch_size=256, subset = "training", validation_split = 0.2, seed =123)
        val_ds = keras.preprocessing.image_dataset_from_directory(
            train_directory, label_mode=label_mode, labels="inferred", shuffle=True, batch_size=256, validation_split=0.2,
            subset="validation", seed=123)

    return val_ds, train_ds


def get_class_weight(zord: str, path: str) -> dict:
    """

    :param zord: zord's name
    :param path: main path
    :return: dict containing the weights that balance data
    """
    folders = data_repartition(zord, path + "/train_set")

    class_weight = weighter(folders)

    print("As data is imbalanced, the following weights will be applied ",
          list(class_weight.values()))

    return class_weight


def augment_data(train_ds, data_augmentation):
    """
    :param train_ds: raw train_ds
    :param data_augmentation: data augmentation sequence
    :return: an augmented training set
    """
    print("Augmenting the train_ds")
    augmented_train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))
    # augmented_train_ds = augmented_train_ds.prefetch(buffer_size=32).shuffle(10)  # facilitates training
    print("Augmentation is done")

    return augmented_train_ds


def build_model(base_model, n: int, zord: str, suffix: str):
    """

    :param base_model: base_model keras model
    :param n: number of outputs
    :param zord: zord's name
    :param suffix: str with base_model id
    :return: a built model
    """

    input_shape = (256, 256, 3)
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.experimental.preprocessing.Rescaling(1.0 / 255.)(inputs)
    x = base_model(x, training=False)
    if suffix != "_inceptionv3":
        x = keras.layers.Reshape((1, -1, 1280))(x)
    x = keras.layers.GlobalAveragePooling2D()(x)

    outputs = keras.layers.Dense(n, activation=nn.softmax)(x)

    model = keras.Model(inputs, outputs)

    model._name = zord

    return model


def get_base_model(base_model: str):
    """
    :param base_model: base_model's name
    :return: base_model
    """
    if base_model == "effnetv2":
        from MegaZord.automl.efficientnetv2 import effnetv2_model
        return effnetv2_model.get_model("efficientnetv2-b0", include_top=False, pretrained=True)

    elif base_model == "effnet":
        from efficientnet.tfkeras import EfficientNetB0
        return EfficientNetB0(weights='imagenet', include_top=False)

    elif base_model == "inceptionv3":
        return keras.applications.InceptionV3(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000,
            classifier_activation="softmax")

    elif base_model == "mobilenetv2":
        return keras.applications.mobilenet_v2.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000,
            classifier_activation="softmax")


    else:
        raise Exception("The model asked is not avalaible")


class SwissKnife:
    """
    Innovative and self explanatory class.
    """

    # zords = ["zord1", "zord2", "zord3", ...]
    def __init__(self, directory: str, base_model: str):
        """
        Checks which zords need to be trained and initiate most attributes
        :param directory: main directory containing zords and data folders
        :param base_model: The model used to do transfer-learning on zords
        """

        assert isinstance(directory, str), "directory parameter must be a str object"
        zords = listdir_nohidden(directory + "/train_set")
        self.zords = []
        self.directory = directory
        self.train_queue = []
        self.labels = []
        self.suffix = "_" + base_model
        self.base_model = get_base_model(base_model)

        for zord in zords:

            print("Checking if " + zord + " model exists...")
            in_file = listdir_nohidden(self.directory + "/train_set/" + zord)

            if len(in_file) == 1:
                print("\tSingle Label Class, no model required.")
                self.labels.append(zord)
                self.zords.append([zord, zord])
            else:
                self.labels.append(in_file)
                self.zords.append([zord, in_file])
                if not os.path.isdir(self.directory + "/zords/" + zord + self.suffix + ".pb"):
                    print(zord + self.suffix + ".pb")
                    self.train_queue.append(zord)
                    print("\t" + zord + " model has not been trained yet. It has been added"
                                        "to the training queue.")

        print("Checking if main_zord exists...")
        if not os.path.isdir(self.directory + "/zords/" + "main_zord" + "_inceptionv3" + ".pb"):
            self.train_queue.append("main_zord")
            print("\tmain_zord" + " model has not been trained yet. It has been added"
                                  "to the training queue.")

        print("\n#############################\n ")
        print("Zord models have been imported, please use .train_zords method to start"
              "the training queue if need "
              "be.\nTraining Queue : ", self.train_queue)

        self.data_augmentation = keras.Sequential([
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomContrast((0, 1))])
        self.labels = flatten(self.labels)

    def train_zords(self, epochs=20):
        """
        Trains the different zords (CNN) in the self.train_queue.
        """
        for zord in self.train_queue:
            val_ds, train_ds = get_data(zord, self.directory)
            class_weight = get_class_weight(zord, self.directory)
            augmented_train_ds = augment_data(train_ds, self.data_augmentation)
            base_model, suffix = (get_base_model("inceptionv3"), "_inceptionv3") if zord == "main_zord" else (
                self.base_model, self.suffix)
            base_model.trainable = False
            model = build_model(base_model, len(class_weight), zord, suffix)

            print("Compilation of the CNN")

            model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])  # the default learning rate is 1e-3

            print("Fit sequence launched...")
            callbacks = [keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)]
            model.fit(augmented_train_ds, epochs=epochs, class_weight=class_weight, validation_data=val_ds,
                      callbacks=callbacks, shuffle=True)

            print("{} zord has been fitted and added to pre_megazord dictionary. \n ".format(zord))
            print("Saving the zord ...")
            model.save(self.directory + "/zords/" + zord + suffix + ".pb")
            del model

        print(
            "Pre Megazord dictionary is now complete. \n You can now fine tune (.fine_tune()) or"
            "call Megazord formation (.assemble_Megazord())")

    def fine_tune(self, zord, fine_tune_at=280, epochs=10, learning_rate=0.0001):
        """
        Fine tunes the zord by his name from the fine_tune_at layer.
        """

        print("_____________ Fine tuning {} __________".format(zord))
        val_ds, train_ds = get_data(zord, self.directory)
        class_weight = get_class_weight(zord, self.directory)
        augmented_train_ds = augment_data(train_ds, self.data_augmentation)

        model = keras.models.load_model(self.directory + "/zords/" + zord + self.suffix + ".pb")

        model.layers[2].trainable = True

        for layer in model.layers[2].layers[1:fine_tune_at]:
            layer.trainable = False

        model.optimizer.learning_rate = learning_rate  # a smaller learning rate is required to fine
        # tune the unfrozen layers
        callbacks = [keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)]
        model.fit(augmented_train_ds, epochs=epochs, class_weight=class_weight, validation_data=val_ds,
                  callbacks=callbacks, shuffle=True)

        print("{} zord has been fine-tuned and added to pre_megazord"
              "dictionary. \n ".format(zord))

        print("Saving the zord ...")
        model.save(self.directory + "/zords/" + zord + self.suffix + ".pb")
        del model

    def assemble_megazord(self):
        """
        Assembles Megazord.
        """

        input_shape = (256, 256, 3)
        inputs = keras.Input(shape=input_shape)

        main_zord = load_model(self.directory + "/zords/" + "main_zord" + "_inceptionv3" + ".pb")
        class_output = main_zord(inputs)
        del main_zord
        # Set to 0 the outputs of the classifier that are not the maximum.
        mask = math.equal(class_output, math.reduce_max(class_output))
        mask = cast(mask, float32)
        transformed_inputs = multiply(mask, class_output)

        nb_class, connected_label_nb, pre_stack = len(self.zords), 0, []

        pre_stack = []
        print(self.zords)
        for zord, labels in tqdm(self.zords):
            if type(labels) == str:
                pre_stack.append(transformed_inputs[0, connected_label_nb])
            else:
                model = load_model(self.directory + "/zords/" + zord + self.suffix + ".pb")
                out = model(inputs)
                for i in range(len(labels)):
                    pre_stack.append(transformed_inputs[0, connected_label_nb] * out[0, i])
                del model
            connected_label_nb += 1

        assert connected_label_nb == nb_class, "Classes are missing"

        print(
            "\n\t\t\t\t\t\t#############################################\n\t\t\t\t\t\t"
            "###########  MEGAZORD DEPLOYED  #############\n\t\t\t\t\t\t################"
            "#############################\n ")

        return keras.Model(inputs, stack([pre_stack], axis=0))

    def save(self, model):
        """
        Saves MegaZord
        """
        print("Saving Megazord")
        model.save(self.directory + "/zords/" + "MegaZord" + self.suffix + ".pb")
        print("Megazord is saved")

    def megazord_to_coreml(self, model):
        """
        Converts Megazord to CoreML.
        """

        print("CoreML conversion is beginning...")

        image_input = ct.ImageType(shape=(1, 256, 256, 3,),
                                   bias=[-1, -1, -1])

        classifier_config = ct.ClassifierConfig(self.labels)

        megazord_cml = ct.convert(model, inputs=[image_input],
                                  classifier_config=classifier_config)

        print("Saving the converted MegaZord...")

        megazord_cml.save(self.directory + "/zords/" + "MegaZord" + self.suffix + ".mlmodel")

        print("Megazord is ready to serve ;)")


if __name__ == "__main__":

    DIRECTORY = "/Volumes/WD_BLACK/ressources"

    complete = True

    swiss_knife = SwissKnife(DIRECTORY, "mobilenetv2")
    swiss_knife.train_zords()

    #for zord in swiss_knife.zords:
        #swiss_knife.fine_tune(zord=zord)

    if complete:
        megazord = swiss_knife.assemble_megazord()

        # swiss_knife.save(MegaZord)
        print(swiss_knife.labels)

        try:
            swiss_knife.megazord_to_coreml(megazord)
        except Exception as e:
            print(e.__class__)

# [o_o]