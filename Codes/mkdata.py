from glob import glob
from tqdm import tqdm
#from rdkit import Chem
import numpy as np
import pandas as pd
import re
import shutil
import os

train = pd.read_csv("../data/train_set.ReorgE.csv")
train.columns = ["index", "SMILES", "Reorg_g", "Reorg_ex"]

max_len = 0 # max_len
error = [] # error가 나는 파일을 검출하기 위해 종종이렇게 처리한다.
g_data, ex_data = [], [] # g_data와 ex_data를 구분해서 저장하낟.
mol = [] # mol_data

for i in tqdm(train[train.columns[0]]): # train의 first column을 돌린다.index가 될 것
	tmp_g = pd.read_csv(f"g_file/{i}" + '_g.csv') # 각각의 index를 통해 파일을 읽는다.
	print(tmp_g["3"])
	# 아마 i에는 train_1234 이런식의 데이터가 들어갈 것이다.
	if len(tmp_g) >= max_len: # data의 max_len을 저장해준다.
		max_len = len(tmp_g)
	print(type(tmp_g))
	for j in tmp_g["3"]: # 딕셔너리 자료형으로 예상되는데 확인해볼 필요가 있다.
		mol.append(j)

mol_list = list(set(mol))
g_data, ex_data = [], []

def get_mol(data):
	if data in mol_list:
		return mol_list.index(data)
	else:
		return -1

for i in tqdm(train[train.columns[0]]):
	tmp_g = pd.read_csv(f"g_file/{i}"+"_g.csv").iloc[:,:4]
	tmp_ex = pd.read_csv(f"ex_file/{i}"+"_ex.csv").iloc[:,:4]

	tmp_g["3"] = tmp_g["3"].apply(lambda x:get_mol(x))
	tmp_ex["3"] = tmp_ex["3"].apply(lambda x :get_mol(x))
	
	tmp_g = np.array(tmp_g)
	tmp_ex = np.array(tmp_ex)
	
	if max_len != len(tmp_g):
		tmp_g = np.concatenate((tmp_g,np.array([[0]*4]*(max_len - tmp_g.shape[0]))))
		tmp_ex = np.concatenate((tmp_ex, np.array([[0]*4]*(max_len-tmp_ex.shape[0]))))
	elif max_len == len(tmp_g):
		tmp_g = tmp_g 
		tmp_ex = tmp_ex

g_data.append(tmp_g)
ex_data.append(tmp_ex)

g_data = np.array(g_data).reshape(len(g_data),-1)
ex_data = np.array(ex_data).reshape(len(ex_data),-1)

train_x = np.concatenate((g_data,ex_data),axis = 1)
train_y = np.array(train.loc[:,["Reorg_g","Reorg_ex"]])

print(train_x.shape)
print(train_y.shape)


from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

LR = MultiOutputRegressor(lgb.LGBMRegressor(random_state=42, n_estimators=1000, max_depth=61, learning_rate=0.075)).fit(train_x, train_y)

if os.path.isdir("test_ex_file"):
	pass
else:
	os.mkdir("test_ex_file")

if os.path.isdir("test_g_file"):
	pass
else:
	os.mkdir("test_g_file")

test_mol = sorted(glob("data/mol_files/test_set/*.mol"))

for i in tqdm(test_mol):
	result = []
	tmp = open(i,'r').read().split("\n")
	for j in tmp:
		tmp_ = re.sub(' +', '',j.lstrip()).split("")
		if len(tmp_)>11:
			result.append(tmp_)

file_name = i.split("/")[-1].split('.')[0]

if "ex" in file_name:
	pd.DataFrame(result).to_csv(f"test_ex_file/{file_name}"+".csv", index = False)
elif "g" in file_name:
	pd.DataFrame(result).to_csv(f"test_g_file/{file_name}"+".csv", index = False)

test = pd.read_csv("../data/test_set.csv")
test_g , test_ex = [], []

for i in tqdm(test[test.columns[0]]):
	tmp_g = pd.read_csv(f"test_g_file/{i}" + "_g.csv").iloc[:,:4]
	tmp_ex = pd.read_csv(f"test_ex_file/{i}"+"_ex.csv").iloc[:,:4]
	tmp_g["3"] = tmp_g["3"].apply(lambda x: get_mol(x))
	tmp_ex["3"] = tmp_ex["3"].apply(lambda x : get_mol(x))

	tmp_g = np.array(tmp_g)
	tmp_ex = np.array(tmp_ex)

	if len(tmp_g) < max_len:
		tmp_g  =np.concatenate((tmp_g, np.array([[0] * 4] * (max_len-tmp_g.shape[0]))))
		tmp_ex = np.concatenate((tmp_ex, np.array([[0] * 4] * (max_len - tmp_ex.shape[0]))))
	elif len(tmp_g) == max_len:
		pass
	else:
		tmp_g = tmp_g[:max_len]
		tmp_ex = tmp_ex[:max_len]

test_g.append(tmp_g)
test_ex.append(tmp_ex)

test_g = np.array(test_g).reshape(len(test_g),-1)
test_ex = np.array(test_ex).reshape(len(test_ex),-1)

test_x = np.concatenate((test_g, test_ex),axis = 1)
print(test_x.shape)

pred = LR.predict(test_x)
submission = pd.read_csv("data/sample_submission.csv")
submission.head()

submission.loc[:,["Reorg_g", "Reorg_ex"]] = pred

submission.to_csv("SAMSUNG_BASELINE.csv",index = False)
