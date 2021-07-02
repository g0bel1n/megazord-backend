import seaborn as sns
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from tensorflow import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utils import zord_from_pb_file, image_from_directory

sns.set_style("dark")
sns.set(rc={'figure.figsize': (10, 10)})


class CrashTest:

    def __init__(self, dir_model):
        test_dir = "/Users/lucas/swiss_knife/data_test"
        self.zord = zord_from_pb_file(dir_model)

        print("loading model...")
        try:
            self.model = keras.models.load_model(dir_model)
            print("sucessful importation")
            print(self.model)
        except OSError:
            print("Importation failed")

        print("loading test dataset...")
        data = image_from_directory(path=test_dir, zord_kind=self.zord)
        self.x_test = data.x
        self.y_test = data.y
        self.labels = data.label_map
        print("data loaded")

    def confmat(self):
        n = 100
        if n > len(self.y_test):
            n = len(self.x_test)
        indexs = random.sample(range(len(self.x_test)), n)
        print(indexs)
        predictions = [self.model.predict((self.x_test[index]).reshape(1, 256, 256, 3)).argmax() for index in
                       tqdm(indexs)]
        y_s = [self.y_test[i] for i in indexs]
        cm = confusion_matrix(y_s, predictions)
        ConfusionMatrixDisplay(confusion_matrix=cm).plot()
        plt.grid(False)
        plt.show()


if __name__ == "__main__":
    test = CrashTest("/Users/lucas/swiss_knife/zords/main_zord.pb")
    test.confmat()
