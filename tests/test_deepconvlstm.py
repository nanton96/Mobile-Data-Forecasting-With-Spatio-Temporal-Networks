from model_architectures.conv_lstm_deep.DeepConvLstm import EF,Encoder,Forecaster,ConvLSTM
from model_architectures.conv_lstm_deep.architecture_specifications import encoder_architecture, forecaster_architecture
###### Define encoder #####
import torch
batch_size = 5
seq_start =12
seq_length = 22
device = torch.cuda.current_device()

x = torch.randn([5,12,100,100]).to(device)
y = torch.randn([5,10,100,100]).to(device)

enc_arch = encoder_architecture(batch_size, device, seq_start)
encoder = Encoder(enc_arch[0],enc_arch[1]).to(device)

###### Define decoder #####
seq_output = seq_length - seq_start

fore_arch = forecaster_architecture(batch_size, device, seq_output)

forecaster=Forecaster(fore_arch[0],fore_arch[1],seq_output).to(device)

model = EF(encoder,forecaster)

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