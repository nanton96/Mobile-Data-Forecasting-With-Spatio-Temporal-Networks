from model_architectures.pred_rnn_pp.PredRNNPP import PredRNNPP
# from model_architectures.conv_lstm_deep.DeepConvLstm import 
import torch 
import utils

device = torch.cuda.current_device()
x = torch.randn([5,12,32,32],requires_grad=False).to(device)
y = torch.randn([5,10,32,32],requires_grad=False).to(device)

num_hidden = [64,64,64,64]

model = PredRNNPP(x.shape,12,10,5,num_hidden,device)
torch.cuda
print(torch.cuda.max_memory_allocated())
model.to(device)
print(torch.cuda.max_memory_allocated())
out = model.forward(x)
print(torch.cuda.max_memory_allocated())
print('out shape' ,out.shape)

se = torch.sum((out - y)**2,(2,3)) # MSE error per frame
loss = torch.mean(se)

print(loss)

loss.backward()
print(torch.cuda.max_memory_allocated())

print('done')
