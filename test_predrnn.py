from model_architectures.pred_rnn_pp.PredRNNPP import PredRNNPP
import torch 
import utils

device = torch.cuda.current_device()
x = torch.randn([5,12,50,50]).to(device)
y = torch.randn([5,10,50,50]).to(device)

num_hidden = [16,32,32,32]

model = PredRNNPP(x.shape,12,10,5,num_hidden,device)

model.to(device)

out = model.forward(x)

print('out shape' ,out.shape)

se = torch.sum((out - y)**2,(2,3)) # MSE error per frame
loss = torch.mean(se)

print(loss)

loss.backward()

print('done')