import plistlib
import pandas as pd
from utils import int_reader

df = pd.read_csv("/Volumes/WD_BLACK/ressources/files/materiel.csv", delimiter=",")
df.set_index(["label"], inplace=True)
root = {}
df["description"]=df["description"].astype(str)
print(df["description"])
for key in df.index:
    label, description, url = (df["Nom appli"].loc[key],
                               "Référence : " + str(df["ref"].loc[key]) +"\nFournisseur : "+
                                                    df["fournisseur"].loc[key] +"\n" +
                                                    ("Description: "+df["description"].loc[key] if not df["description"].loc[key]=="nan" else ""),
                               df["url"].loc[key]) if key in df.index else (key, "", "")

    temp_dic = {"label": label, "description": description, "url" : url}
    root[key] = temp_dic

with open("ProductCatalog.plist", 'wb') as fp:
    plistlib.dump(root, fp)

# [o_o]