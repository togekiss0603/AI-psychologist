#加载词向量
from tqdm import tqdm
f=open('embedding.txt',"r",encoding="utf-8")
vectorList=f.readlines()
word_to_id,id_to_vector={},[]
n=int(vectorList[0].split(' ')[0])#数量
v=int(vectorList[0].split(' ')[1])#词向量长度
word_to_id[' ']=0
id_to_vector.append([.0]*v)
for x in tqdm(range(n)):
    i=x+1
    word=vectorList[i].split(" ",1)[0]
    vector = list(m.ap(float,vectorList[i].split(" ",1)[1].strip().split(" ")))
    word_to_id[word]=i
    id_to_vector.append(vector)
word_to_id['null']=n+1
id_to_vector.append([.1]*v)

import torch

class basic_param:
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')  # 是否使用gpu
    # device=torch.device('cpu')
    MAXLENGTH = 50
    train_range = 12000  # 训练集

    batch_size_train = 128  # 训练时的batch size
    batch_size_val = 128  # 验证集的batch size
    lr = 0.0001

    hidden_size = 64  # 隐藏层大小
    num_layers = 8  # 层数
    categoryCnt = 5  # 分类数
    V = v

from model import NLPNet
model = NLPNet(id_to_vector, basic_param)
state_dict = torch.load('best_model.pt', map_location='cpu')
model.load_state_dict(state_dict)
model.to(basic_param.device)

model.eval()
s = ""
with torch.no_grad():
    pred_Pr = model(s)
    pred = torch.argmax(pred_Pr, dim=1)
    print(pred)

from pyhanlp import *
s=""
disease_list=['抑郁症','焦虑症','双相情感障碍','创伤后应激障碍','恐慌症','厌食症','暴食症']
disease=""
seg_s=HanLP.segment(s)
for i in seg_s:
    if i in disease_list:
        disease=i
        break
if disease=='':#如果没有识别到疾病
    disease=diseaseReg(s)