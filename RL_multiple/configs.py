class Strategy_Configs:
    def __init__(self):
        self.task_name = 'classification'
        self.seq_len = 20
        self.label_len = 1
        self.pred_len = 1
        self.enc_in = 9  # number of input features
        self.dec_in = 9  # number of input features
        self.c_out = 1  # number of output features
        self.d_model = 128
        self.embed = 'timeF'
        self.freq = 'd'
        self.dropout = 0.1
        self.top_k = 5
        self.num_kernels = 6
        self.d_ff = 512
        self.e_layers = 2
        self.output_attention = False
        self.num_class = 5
class Allocation_Configs:
    def __init__(self):
        self.task_name = 'classification'
        self.seq_len = 20
        self.label_len = 1
        self.pred_len = 1
        self.enc_in = 9*6  # number of input features
        self.dec_in = 9*6  # number of input features
        self.c_out = 1  # number of output features
        self.d_model = 128
        self.embed = 'timeF'
        self.freq = 'd'
        self.dropout = 0.1
        self.top_k = 5
        self.num_kernels = 6
        self.d_ff = 512
        self.e_layers = 2
        self.output_attention = False
        self.num_class = 6