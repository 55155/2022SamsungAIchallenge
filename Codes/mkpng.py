'''
from rdkit import Chem
from rdkit.Chem import rdAbbreviations,AllChem
import IPython.display

abbrevs = rdAbbreviations.ParseAbbreviations("CH<sub>3</sub> [Ch3]")
m = Chem.MolFromSmiles('[H]N[C@@H](CCCCN)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](C([H])([H])[H])C(=O)N[C@@H](CC1=CNC2=C1C=CC=C2)C(O)=O')
m = rdAbbreviations.CondenseMolAbbreviations(m,abbrevs)
template= Chem.MolFromSmiles("C(=O)CNC(=O)CNC(=O)CNC(=O)CCCC")
AllChem.Compute2DCoords(template)
AllChem.GenerateDepictionMatching2DStructure(m,template)
'''
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import os
import pandas as pd
from tqdm import tqdm
from PIL import Image

# 1. 데이터로 사진만 주었을 때 계산 해보기
train = pd.read_csv("/Users/bangseongjin/Desktop/2022SamsungAIchallenge/data/train_set.ReorgE.csv") # 절대경로
train.columns = ["index", "SMILES", "Reorg_g", "Reorg_ex"]
print(len(train))
print(train.head())


if not os.path.isdir("png_g"):
    os.mkdir("png_g")
else:
    pass
'''
for i in tqdm(range(len(train))):
    m = AllChem.MolFromSmiles(train["SMILES"][i])
    drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
    drawer.drawOptions().explicitMethyl = True
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, m)
    drawer.FinishDrawing()
    # binary 쓰기 전용

    with open(f'png_g/{train["index"][i]}.png', 'wb') as f:
        f.write(drawer.GetDrawingText())
'''
'''
for i in tqdm(range(len(train))):
    im = Image.open(f"/Users/bangseongjin/Desktop/2022SamsungAIchallenge/Codes/png_g/train_{i}"+".png").convert("L")
    im.save(f"/Users/bangseongjin/Desktop/2022SamsungAIchallenge/Codes/png_g/train_{i}.png")
'''
test = pd.read_csv("/Users/bangseongjin/Desktop/2022SamsungAIchallenge/data/test_set.csv")
test.columns = ["index", "SMILES"]

if not os.path.isdir("png_test"):
    os.mkdir("png_test")
else:
    pass

for i in tqdm(range(len(test))):
    m = AllChem.MolFromSmiles(test["SMILES"][i])
    drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
    drawer.drawOptions().explicitMethyl = True
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, m)
    drawer.FinishDrawing()
    # binary 쓰기 전용

    with open(f'png_test/{test["index"][i]}.png', 'wb') as f:
        f.write(drawer.GetDrawingText())

for i in tqdm(range(len(test))):
    im = Image.open(f"/Users/bangseongjin/Desktop/2022SamsungAIchallenge/Codes/png_test/test_{i}.png").convert("L")
    im.save(f"/Users/bangseongjin/Desktop/2022SamsungAIchallenge/Codes/png_test/test_{i}.png")

# normalization

'''

greyImage = "/Users/bangseongjin/Desktop/2022SamsungAIchallenge/Codes/png_test/test_0.png"
normalizedImage = uint8(255*mat2gray(grayImage));
imshow(normalizedImage);
'''