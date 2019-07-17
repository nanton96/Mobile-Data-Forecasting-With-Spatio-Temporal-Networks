from model_architectures.pred_rnn_pp.PredRNNPP_less_mem import PredRNNPP
# from model_architectures.conv_lstm_deep.DeepConvLstm import 
import torch 
import utils

device = torch.cuda.current_device()
x = torch.randn([5,12,50,50]).to(device)
y = torch.randn([5,10,50,50]).to(device)

num_hidden = [16,32,32]
halve_dim = [True,True,False]
strides = [2,2,1]
padding = [1,0,1]
kernel_sizes = [4,3,3]

model = PredRNNPP(x.shape,12,10,5,num_hidden,strides,padding,kernel_sizes,halve_dim,device)
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