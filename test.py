from megazord_init import *
import os


def test_listdir_nohidden_3(dir = "data_test"):
    assert listdir_nohidden(dir, jpg_only=False) == ['ball_bearing', 'handle', 'wheel']

def test_listdir_nohidden_0(dir="data_test"):
    assert listdir_nohidden(dir, jpg_only=True) == []

def test_data_repartition_zord(dir = "data_test/wheel"):
    assert data_repartition("wheel", dir) == [2,3,2]


def test_data_repartition_main_zord(dir="data_test"):
    assert data_repartition("main_zord", dir) == [6,5,7]

def test_weighter():
    folders = [1,5,9]
    assert weighter(folders) == {0: 9.0, 1: 1.8, 2: 1.0}







