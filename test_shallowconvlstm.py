
from model_architectures.conv_lstm_shallow.ShallowConvLstm import ConvLSTMModel
import torch
batch_size = 5
seq_start =12
seq_length = 22
device = torch.cuda.current_device()

x = torch.randn([5,12,100,100]).to(device)
y = torch.randn([5,10,100,100]).to(device)

seq_output = seq_length - seq_start

model = ConvLSTMModel(
            input_size = x.shape[-1],
            seq_start = seq_start, 
            seq_length = seq_length, 
            batch_size = batch_size,
            use_gpu=True)

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