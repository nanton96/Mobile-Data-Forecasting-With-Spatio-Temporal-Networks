
def create_model(model,args,device):
    
    if model.lower()=='shallowconvlstm':
        from model_architectures.conv_lstm_shallow.ShallowConvLstm import ConvLSTMModel
        return ConvLSTMModel(
            input_size = args.image_height,
            seq_start = args.seq_start, 
            seq_length = args.seq_length, 
            batch_size = args.batch_size,
            use_gpu=args.use_gpu)
    elif model.lower()=='deepconvlstm':

        from model_architectures.conv_lstm_deep.DeepConvLstm import EF,Encoder,Forecaster,ConvLSTM
        from model_architectures.conv_lstm_deep.architecture_specifications import encoder_architecture, forecaster_architecture
        ###### Define encoder #####
        
        enc_arch = encoder_architecture(args.batch_size, device, args.seq_start)
        encoder = Encoder(enc_arch[0],enc_arch[1]).to(device)
        
        ###### Define decoder #####
        seq_output = args.seq_length - args.seq_start

        fore_arch = forecaster_architecture(args.batch_size, device, seq_output)
        
        forecaster=Forecaster(fore_arch[0],fore_arch[1],seq_output).to(device)

        model = EF(encoder,forecaster)
        return model
        
    elif model.lower()=='predrnnpp':
        from model_architectures.pred_rnn_pp.PredRNNPP import PredRNNPP

        num_hidden = [4,8,8,8]
        input_shape = [args.batch_size,args.seq_start,args.image_height,args.image_height]
        model = PredRNNPP(input_shape,args.seq_start,args.seq_length-args.seq_start,args.batch_size,num_hidden,device)
        return model

    else:
        raise ValueError('model ' + model + ' not implemented')