from consult.consult_scene_model import MyTransformerModel
from torchtext.legacy import data
from torchtext.vocab import Vectors
import torch
import jieba
import tqdm
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def tokenizer(text):
    return [word for word in jieba.lcut(text) if word.strip()]

def get_dataset(sentence,text_field, label_field):
    fields = [('text', text_field), ('label', label_field)]
    examples = []
    examples.append(data.Example.fromlist([ sentence, None], fields))
    return examples,fields

class Consult_Main():
    def __init__(self):
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        # 使用jieba自定义分词函数
        print("a")
        self.EMBEDDING_DIM = 300       # 词向量维度
        print("b")
        # 两个Field对象定义字段的处理方法（文本字段、标签字段）
        self.TEXT = data.Field(tokenize=tokenizer, batch_first=True, lower=True) # 是否Batch_first. 默认值: False.
        self.LABEL = data.LabelField(dtype=torch.float)
        print("c")

        self.vectors = Vectors(name='consult/sgns.merge.word')
        self.model=MyTransformerModel(5002,self.EMBEDDING_DIM,p_drop=0.5, h=2, output_size=4).to(self.device)
        print("d")
        self.model.load_state_dict(torch.load("consult/wordavg-modelbetter1.pt"))
        print("e")

    def __call__(self,sentence):
        exam,fild=get_dataset(sentence,self.TEXT,self.LABEL)
        x=data.Dataset(exam,fild)
        self.TEXT.build_vocab(x, max_size=550, vectors=self.vectors)
        self.LABEL.build_vocab(x)
        iterator = data.BucketIterator(x,batch_size=1,device=self.device,sort=False,sort_within_batch=False,sort_key=lambda x: len(x.text))
        with torch.no_grad():
            for batch in iterator:
                mask = 1 - (batch.text == self.TEXT.vocab.stoi['<pad>']).float()
                pred = torch.argmax(self.model(batch.text, mask))
        return pred.cpu().item()

if __name__ == "__main__":
    model = Consult_Main()

    print(model("和爸爸妈妈吵架了"))
    print(model('和老公吵架了'))