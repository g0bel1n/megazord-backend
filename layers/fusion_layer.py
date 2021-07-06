from tensorflow.keras.layers import Layer
from tensorflow import math, multiply, cast, float32, stack


class ZordsCombination(Layer):

    def __init__(self, zords):
        super(ZordsCombination, self).__init__()
        self.zords = zords

    def call(self, inputs, *args, **kwargs):

        class_output = self.zords["main_zord"][0](inputs)

        # Set to 0 the outputs of the classifier that are not the maximum.
        mask = math.equal(class_output, math.reduce_max(class_output))
        mask = cast(mask, float32)
        transformed_inputs = multiply(mask, class_output)

        nb_class, connected_label_nb, pre_stack = len(self.zords["main_zord"][1]), 0, []

        for key in self.zords:
            if key != "main_zord":
                zorg_plus = self.zords[key]
                if len(zorg_plus[1]) == 1:
                    pre_stack.append(transformed_inputs[0, connected_label_nb])
                else:
                    out = zorg_plus[0](inputs)
                    for i in range(len(zorg_plus[1])):
                        pre_stack.append(transformed_inputs[0, connected_label_nb] * out[0, i])
                connected_label_nb += 1
        assert connected_label_nb == nb_class, "Classes are missing"

        return stack([pre_stack], axis=0)

