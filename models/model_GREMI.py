import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_model import define_act_layer


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x

class Fusion(nn.Module):
    def __init__(self, num_class, num_views, hidden_dim, dropout, in_dim, dim1, dim2, dim3, alpha=0.5):
        super().__init__()
        self.gat1 = GAT(dropout=0.5, alpha=alpha, dim=dim1)
        self.gat2 = GAT(dropout=0.5, alpha=alpha, dim=dim2)
        self.gat3 = GAT(dropout=0.5, alpha=alpha, dim=dim3)

        self.views = len(in_dim)
        self.classes = num_class
        self.dropout = dropout
        self.hidden_dim = hidden_dim

        self.FeatureInforEncoder = nn.ModuleList([LinearLayer(in_dim[view], in_dim[view]) for view in range(self.views)])
        self.TCPConfidenceLayer = nn.ModuleList([LinearLayer(hidden_dim[0], 1) for _ in range(self.views)])
        self.TCPClassifierLayer = nn.ModuleList([LinearLayer(hidden_dim[0], num_class) for _ in range(self.views)])

        self.MMClasifier = []
        for layer in range(1, len(hidden_dim) - 1):
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[0], hidden_dim[layer]))
            self.MMClasifier.append(nn.ReLU())
            self.MMClasifier.append(nn.Dropout(p=dropout))
        if len(self.MMClasifier):
            self.MMClasifier.append(LinearLayer(hidden_dim[-1], num_class))
        else:
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[-1], num_class))
        self.MMClasifier = nn.Sequential(*self.MMClasifier)


    def forward(self, omic1, omic2, omic3, adj1, adj2, adj3, label=None, infer=False):
        # print(omic1.shape, omic2.shape, omic3.shape)
        # print(adj1.shape, adj2.shape, adj3.shape)
        output1, gat_output1 = self.gat1(omic1, adj1)
        output2, gat_output2 = self.gat2(omic2, adj2)
        output3, gat_output3 = self.gat3(omic3, adj3)
        #
        feature = dict()
        feature[0], feature[1], feature[2] = output1, output2, output3
        #
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        loss_function = nn.CrossEntropyLoss()
        #
        FeatureInfo, TCPLogit, TCPConfidence = dict(), dict(), dict()
        for view in range(self.views):
            feature[view] = F.relu(feature[view])
            feature[view] = F.dropout(feature[view], self.dropout, training=self.training)
            TCPLogit[view] = self.TCPClassifierLayer[view](feature[view])
            TCPConfidence[view] = self.TCPConfidenceLayer[view](feature[view])
            feature[view] = feature[view] * TCPConfidence[view]

        MMfeature = torch.cat([i for i in feature.values()], dim=1)
        MMlogit = self.MMClasifier(MMfeature)
        if infer:
            return MMlogit
        MMLoss = torch.mean(criterion(MMlogit, label))
        loss_gat1 = loss_function(gat_output1,label)
        loss_gat2 = loss_function(gat_output2,label)
        loss_gat3 = loss_function(gat_output3,label)
        gat_loss = dict()
        gat_loss[0], gat_loss[1], gat_loss[2] = loss_gat1, loss_gat2, loss_gat3
        for view in range(self.views):
            MMLoss = MMLoss + gat_loss[view]
            pred = F.softmax(TCPLogit[view], dim=1)
            p_target = torch.gather(input=pred, dim=1, index=label.unsqueeze(dim=1)).view(-1)
            confidence_loss = torch.mean(
                F.mse_loss(TCPConfidence[view].view(-1), p_target) + criterion(TCPLogit[view], label))
            MMLoss = MMLoss + confidence_loss
        return MMLoss, MMlogit, gat_output1, gat_output2, gat_output3, output1, output2, output3

    def infer(self, omic1, omic2, omic3, adj1, adj2, adj3):
        with torch.no_grad():  # Prevent gradient storage
            MMlogit = self.forward(omic1, omic2, omic3, adj1, adj2, adj3, infer=True)
            result = MMlogit.detach().clone()  # Clone to detach from computation graph
    
        # Explicitly clean up tensors
        del MMlogit
        torch.cuda.empty_cache()  # Force CUDA memory cleanup if using GPU
    
        return result


class GAT(nn.Module):
    def __init__(self, dropout, alpha, dim, input_dim=700):

        super(GAT, self).__init__()
        self.dropout = dropout
        self.act = define_act_layer(act_type='none')
        self.dim = dim
        self.nhids = [8, 16, 12]
        self.nheads = [4, 3, 4]
        self.fc_dim = [600, 256, 64, 32]

        self.attentions1 = [GraphAttentionLayer(
            1, self.nhids[0], dropout=dropout, alpha=alpha, concat=True) for _ in range(self.nheads[0])]
        for i, attention1 in enumerate(self.attentions1):
            self.add_module('attention1_{}'.format(i), attention1)

        self.attentions2 = [GraphAttentionLayer(
            self.nhids[0] * self.nheads[0], self.nhids[1], dropout=dropout, alpha=alpha, concat=True) for _ in
            range(self.nheads[1])]
        for i, attention2 in enumerate(self.attentions2):
            self.add_module('attention2_{}'.format(i), attention2)

        self.attentions3 = [GraphAttentionLayer(
            self.nhids[1] * self.nheads[1], self.nhids[2], dropout=dropout, alpha=alpha, concat=True) for _ in
            range(self.nheads[2])]
        for i, attention3 in enumerate(self.attentions3):
            self.add_module('attention3_{}'.format(i), attention3)

        self.dropout_layer = nn.Dropout(p=self.dropout)


        self.pool1 = torch.nn.Linear(self.nhids[0] * self.nheads[0], 1)
        self.pool2 = torch.nn.Linear(self.nhids[1] * self.nheads[1], 1)
        self.pool3 = torch.nn.Linear(self.nhids[2] * self.nheads[2], 1)

        lin_input_dim = 3 * self.dim
        self.fc1 = nn.Sequential(
            nn.Linear(lin_input_dim, self.fc_dim[0]),
            nn.ELU(),
            nn.AlphaDropout(p=self.dropout, inplace=False))
        self.fc1.apply(xavier_init)

        self.fc2 = nn.Sequential(
            nn.Linear(self.fc_dim[0], self.fc_dim[1]),
            nn.ELU(),
            nn.AlphaDropout(p=self.dropout, inplace=False))
        self.fc2.apply(xavier_init)

        self.fc3 = nn.Sequential(
            nn.Linear(self.fc_dim[1], self.fc_dim[2]),
            nn.ELU(),
            nn.AlphaDropout(p=self.dropout, inplace=False))
        self.fc3.apply(xavier_init)

        self.fc4 = nn.Sequential(
            nn.Linear(self.fc_dim[2], self.fc_dim[3]),
            nn.ELU(),
            nn.AlphaDropout(p=self.dropout, inplace=False))
        self.fc4.apply(xavier_init)

        self.fc5 = nn.Sequential(
            nn.Linear(self.fc_dim[3], 4))
        self.fc5.apply(xavier_init)

    def forward(self, x, adj):


        x0 = torch.mean(x, dim=-1)
        x = self.dropout_layer(x)
        x = torch.cat([att(x, adj) for att in self.attentions1], dim=-1)

        x1 = self.pool1(x).squeeze(-1)
        x = self.dropout_layer(x)
        x = torch.cat([att(x, adj) for att in self.attentions2], dim=-1)

        x2 = self.pool2(x).squeeze(-1)
        x = torch.cat([x0, x1, x2], dim=1)

        x = self.fc1(x)
        x = self.fc2(x)
        x1 = self.fc3(x)
        x = self.fc4(x1)
        x = self.fc5(x)

        output = x1
        gat_output = x

        return output, gat_output


class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(p=self.dropout)

    def forward(self, input, adj):
        """
        input: mini-batch input. size: [batch_size, num_nodes, node_feature_dim]
        adj:   adjacency matrix. size: [num_nodes, num_nodes].  need to be expanded to batch_adj later.
        """
        h = torch.matmul(input, self.W)
        bs, N, _ = h.size()

        attention = torch.zeros(bs, N, N, device=h.device)
        chunk_size = 50  # Adjust based on your memory constraints
        
        for i in range(0, N, chunk_size):
            end_i = min(i + chunk_size, N)
            
            # Compute attention for this chunk
            for j in range(0, N, chunk_size):
                end_j = min(j + chunk_size, N)
                
                # Get features for nodes i and j
                h_i = h[:, i:end_i, :]  # [bs, chunk, out_features]
                h_j = h[:, j:end_j, :]  # [bs, chunk, out_features]
                
                # Compute attention coefficients
                h_i_expanded = h_i.unsqueeze(2).expand(-1, -1, end_j-j, -1)  # [bs, chunk_i, chunk_j, out_features]
                h_j_expanded = h_j.unsqueeze(1).expand(-1, end_i-i, -1, -1)  # [bs, chunk_i, chunk_j, out_features]
                
                # Concatenate features
                a_input = torch.cat([h_i_expanded, h_j_expanded], dim=-1)  # [bs, chunk_i, chunk_j, 2*out_features]
                
                # Compute attention scores
                e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))  # [bs, chunk_i, chunk_j]
                
                # Apply adjacency mask
                adj_chunk = adj[i:end_i, j:end_j].unsqueeze(0).expand(bs, -1, -1)  # [bs, chunk_i, chunk_j]
                
                # Set attention to zero for non-connected nodes
                zero_vec = -9e15 * torch.ones_like(e)
                e_masked = torch.where(adj_chunk > 0, e, zero_vec)
                
                # Store in the attention matrix
                attention[:, i:end_i, j:end_j] = e_masked
        
        # Apply softmax along the last dimension
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout_layer(attention)
        
        # Compute output features using attention weights
        h_prime = torch.bmm(attention, h)  # [bs, N, out_features]
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'