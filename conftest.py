from .utilitaries.utils import listdir_nohidden, data_repartition, weighter, diver


def test_listdir_nohidden_3(path="data_test"):
    assert listdir_nohidden(path, jpg_only=False) == ['ball_bearing', 'handle', 'wheel']


def test_listdir_nohidden_0(path="data_test"):
    assert listdir_nohidden(path, jpg_only=True) == []


def test_data_repartition_zord(path="data_test"):
    assert data_repartition("wheel", path) == [2, 3, 2]


def test_data_repartition_main_zord(path="data_test"):
    assert data_repartition("main_zord", path) == [6, 6, 7]


def test_weighter():
    folders = [1, 5, 9]
    assert weighter(folders) == {0: 9.0, 1: 1.8, 2: 1.0}


def test_diver():
    assert diver("data_test/ball_bearing") == "data_test/ball_bearing/ball_bearing/ball_bearing"

"""
def test_zord_combination():
    inputs = np.empty((1, 256, 256, 3))

    def mock_main_zord(inputs):
        return np.array([[0.5, 0.2, 0.1, 0.2]])

    def mock_zord1(inputs):
        return np.array([[0.4, 0.6]])

    def mock_zord2(inputs):
        return np.array([[0.9, 0.1]])

    zords = {"main_zord": [mock_main_zord, [0, 1, 2, 3]],
             "zord1": [mock_zord1, [5, 6]],
             "zord2": [mock_zord2, [7, 8]],
             "zord3": [1, [1]], "zord4": [1, [1]]}

    value_to_test = fusion_layer.ZordsCombination(zords)(inputs)
    true_value = np.array([[0.2, 0.3, 0, 0, 0, 0]], dtype="float32")

    assert not np.any(value_to_test.numpy() - true_value)
"""