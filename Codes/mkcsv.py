from glob import glob
from tqdm import tqdm
#from rdkit import Chem
import numpy as np
import pandas as pd
import re
import shutil
import os

train = pd.read_csv("../data/train_set.ReorgE.csv")
train.columns = ["index","SMILES","Reorg_g","Reorg_ex"]

print(train.head())

if os.path.isdir("ex_file"):
	pass # if ex_file directory exist pass
else:
	os.mkdir("ex_file") # else mkdir ex_file

if os.path.isdir("g_file"):
	pass
else:
	os.mkdir("g_file") # same process

train_mol = sorted(glob("../data/mol_files/train_set/*.mol"))
#.mol 확장자를 가지고 있는 파일들을 전부 모아서 하나의 리스트로 만드는데, 이때 파일 정렬을 해준다. 하나의 리스트가 될 예정

print(type(train_mol)) # type print ==> list

for i in tqdm(train_mol):  # tqdm은 프로세스 바를 의미한다.
	result = []
	tmp = open(i,'r').read().split("\n") # 읽기  전용으로 읽고 해당 파일 전체를 읽되, 줄바꿈으로 구분하여 저장한다.
	for j in tmp: # .mol 의 데이터들을 돌리면서
		
		tmp_ = re.sub(' +', ' ', j.lstrip()).split(' ') # 데이터 정리 과정
		# 왼쪽 공백이 없어진 .mol파일에서 정규표현식으로 " +" --> ' ' 로 바꾼다. 그리고 탭으로 구분한 리스트를 만들어준다. 
		if len(tmp_) > 11: # 정리를 하고 난 tmp_ 의 길이가 11이 넘어가면 특정 데이터라는 뜻인데,
            		result.append(tmp_) # 이 데이터만을 result 라는 빈공간 리스트에 담는다.
	
	file_name = i.split('/')[-1].split('.')[0]
	print(file_name)    

	if "ex" in file_name:
		pd.DataFrame(result).to_csv(f"ex_file/{file_name}" + ".csv", index = False)
	elif "g" in file_name:
		pd.DataFrame(result).to_csv(f"g_file/{file_name}" + ".csv", index = False)
