import plistlib
import pandas as pd

df = pd.read_csv("/Volumes/WD_BLACK/ressources/files/materiel.csv", delimiter=",")
df.set_index(["label"], inplace=True)
root = {}
print(df)
for index in df.index:
    root[str(df["ref"].loc[index])] = index

with open("refToId.plist", 'wb') as fp:
    plistlib.dump(root, fp)

# [o_o]