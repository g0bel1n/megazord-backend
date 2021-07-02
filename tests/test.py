from megazord_project.utilitaries.utils import *


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
    assert diver("../data_test/ball_bearing") == "data_test/ball_bearing/ball_bearing/ball_bearing"
