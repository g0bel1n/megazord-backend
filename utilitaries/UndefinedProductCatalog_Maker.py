import plistlib
import pandas as pd
from utils import int_reader

df = pd.read_csv("/Volumes/WD_BLACK/ressources/files/UndefinedProductCatalog.csv", delimiter=",")
print(df.columns)
df.set_index(["piece"], inplace=True)
root = {}
for key in df.index:
    answer1, answer2, question = df["answer1"].loc[key], df["answer2"].loc[key], df["question"].loc[key]
    sol1,sol2 = df["sol1"].loc[key], df["sol2"].loc[key]

    temp_dic = {"answer1": answer1, "answer2": answer2, "question" : question, "sol1": sol1, "sol2": sol2}
    root[key] = temp_dic

with open("UndefinedProductCatalog.plist", 'wb') as fp:
    plistlib.dump(root, fp)

# [o_o]