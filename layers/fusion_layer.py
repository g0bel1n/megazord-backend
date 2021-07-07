from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model
from tensorflow import math, multiply, cast, float32, stack
from tqdm import tqdm


class ZordsCombination(Layer):

    def __init__(self, zords, directory, suffix):
        super(ZordsCombination, self).__init__()
        self.zords = zords
        self.directory = directory
        self.suffix=suffix

    def call(self, inputs, *args, **kwargs):

        main_zord = load_model(self.directory+ "/zords/"+ "main_zord" + self.suffix + ".pb")
        class_output = main_zord(inputs)
        del main_zord
        # Set to 0 the outputs of the classifier that are not the maximum.
        mask = math.equal(class_output, math.reduce_max(class_output))
        mask = cast(mask, float32)
        transformed_inputs = multiply(mask, class_output)

        nb_class, connected_label_nb, pre_stack = len(self.zords), 0, []

        pre_stack = []

        for zord, labels in tqdm(self.zords):
            if len(labels) == 1:
                pre_stack.append(transformed_inputs[0, connected_label_nb])
            else:
                model = load_model(self.directory+ "/zords/"+ zord + self.suffix + ".pb")
                out = model(inputs)
                for i in range(len(labels)):
                    pre_stack.append(transformed_inputs[0, connected_label_nb] * out[0, i])
            connected_label_nb += 1
        assert connected_label_nb == nb_class, "Classes are missing"

        return stack([pre_stack], axis=0)

