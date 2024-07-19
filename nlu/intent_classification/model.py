import torch
class NLPNet(torch.nn.Module) : #模型：
    def __init__(self,id_to_vector,basic_param):
        super(NLPNet, self).__init__()
        #一层embedding
        self.embedding = torch.nn.Embedding.from_pretrained(
            torch.tensor(id_to_vector)
        )
        self.embedding.weight.requires_grad = True
        #num_layers层
        self.inputSize = basic_param.V
        self.lstm = torch.nn.LSTM(
            self.inputSize, 
            basic_param.hidden_size//2, 
            num_layers = basic_param.num_layers,
            bidirectional = True, 
            batch_first=True
        )
        #一个线性层，整合RNN（LSTM）提取的信息
        self.linear1 = torch.nn.Linear(
            basic_param.hidden_size,
            self.inputSize
        )
        # 用一个简化的attention层，将信息综合
        self.chooseWeight = torch.randn(
            [basic_param.MAXLENGTH,1],
            requires_grad=True,
            device = basic_param.device
        )
        # 再一层线性层，将输出变为预测维度(2)
        self.linear2 = torch.nn.Linear(
            self.inputSize,
            basic_param.categoryCnt
        )
    def forward(self,x) : # 预测
        x = self.embedding(x) #[batch*length*V]
        lstm_output, (hn, cn) = self.lstm(x) 
        output = self.linear1(lstm_output) #batch*length*categ -> batch*length*inputSize
        attention = (output*self.chooseWeight).sum(dim=1) #batch*inputSize\
        return self.linear2(attention)
    