import plistlib
import numpy as np
from  utils import int_reader

dict = int_reader("label")
root = {}
for key in dict:
    temp_dic = {"label" : key, "description": "bla bla bla"}
    root[key] = temp_dic


with open("ProductCatalog.plist", 'wb') as fp:
    plistlib.dump(root, fp)
