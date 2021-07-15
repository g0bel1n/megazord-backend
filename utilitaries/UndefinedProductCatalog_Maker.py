import plistlib
import pandas as pd
from utils import int_reader

df = pd.read_csv("/Users/lucas/swiss_knife/UndefinedProd.csv", delimiter=";")
print(df.columns)
df.set_index(["class"], inplace=True)
root = {}
for key in df.index:
    right, left, question = df["right"].loc[key], df["left"].loc[key], df["question"].loc[key],

    temp_dic = {"right": right, "left": left, "question" : question}
    root[key] = temp_dic

with open("UndefinedProductCatalog.plist", 'wb') as fp:
    plistlib.dump(root, fp)
