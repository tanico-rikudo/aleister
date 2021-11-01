class SimpleDnn(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(SimpleDnn, self).__init__()
        
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, layer_dim)
        self.layer_3 = nn.Linear(layer_dim, layer_dim//2)
        self.layer_out = nn.Linear(layer_dim//2, output_dim) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(hidden_dim)
        self.batchnorm2 = nn.BatchNorm1d(layer_dim)
        # self.batchnorm3 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        x = self.layer_1(x)
        # x = self.batchnorm1(x)
        # x = self.relu(x)
        
        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_3(x)
        # x = self.batchnorm3(x)
        # x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_out(x)
        
        return x