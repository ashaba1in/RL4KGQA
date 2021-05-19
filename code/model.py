from torch import nn
import torch

import numpy as np
from drive.MyDrive.Diploma.load_config import config


class RelativePosition(nn.Module):
    def __init__(self, hid_dim, k):
        super().__init__()

        self.hid_dim = hid_dim
        self.k = k
        self.embeddings = nn.Parameter(torch.zeros(k * 2 + 1, hid_dim))
        nn.init.xavier_uniform_(self.embeddings)

    def forward(self, q_len, k_len):
        device = next(self.parameters()).device

        r_pos = torch.arange(k_len)[None, :] - torch.arange(q_len)[:, None]
        r_pos_clip = torch.clamp(r_pos, -self.k, self.k) + self.k
        r_pos_clip = torch.LongTensor(r_pos_clip).to(device)
        embeddings = self.embeddings[r_pos_clip].to(device)

        return embeddings


class MultiHeadAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, max_relative_pos=90):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.scale = np.sqrt(self.head_dim)

        self.dropout = nn.Dropout(dropout)

        # self.relative_pos_k = RelativePosition(self.head_dim, max_relative_pos)
        # self.relative_pos_v = RelativePosition(self.head_dim, max_relative_pos)

        self.fc_out = nn.Linear(hid_dim, hid_dim)

    def forward(self, query, key, value, mask):
        batch_size = query.shape[0]

        Q = self.fc_q(query).view(batch_size, -1, self.n_heads, self.head_dim)
        K = self.fc_k(key).view(batch_size, -1, self.n_heads, self.head_dim)
        V = self.fc_v(value).view(batch_size, -1, self.n_heads, self.head_dim)

        q_len = Q.shape[1]
        k_len = K.shape[1]
        v_len = V.shape[1]

        r_Q = Q.permute(1, 0, 2, 3).contiguous()
        r_Q = r_Q.view(q_len, batch_size * self.n_heads, self.head_dim)

        # r_K = self.relative_pos_k(q_len, k_len)

        # r_energy = torch.einsum('qih,qkh->iqk', r_Q, r_K)
        # r_energy = r_energy.view(batch_size, self.n_heads, q_len, k_len)

        energy = torch.einsum('bqnh,bknh->bnqk', Q, K)

        energy = (energy) / self.scale
        energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.dropout(torch.softmax(energy, dim=-1))

        # r_V = self.relative_pos_v(q_len, v_len)
        # r_x = attention.permute(2, 0, 1, 3).contiguous()
        # r_x = r_x.view(q_len, batch_size * self.n_heads, k_len)
        # r_x = torch.einsum('qik,qkh->iqh', r_x, r_V)
        # r_x = r_x.view(batch_size, self.n_heads, q_len, self.head_dim)
        # r_x = torch.einsum('bnqh->bqnh', r_x).view(batch_size, q_len, self.hid_dim)

        x = torch.einsum('bnqk,bknh->bqnh', attention, V)
        x = torch.einsum('bnqh->bqnh', x).view(batch_size, q_len, self.hid_dim)

        # x += r_x

        x = self.fc_out(x)

        return x


class Feedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.swish = lambda x: x * torch.sigmoid(x)
        self.dropout = nn.Dropout(dropout)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

    def forward(self, x):
        x = self.swish(self.fc_1(x))
        x = self.fc_2(self.dropout(x))
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout):
        super().__init__()

        self.attention = MultiHeadAttention(hid_dim, n_heads, dropout)
        self.attention_ln = nn.LayerNorm(hid_dim)

        self.feedforward = Feedforward(hid_dim, pf_dim, dropout)
        self.feedforward_ln = nn.LayerNorm(hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        shortcut = src
        src = self.attention(src, src, src, src_mask)
        src = self.attention_ln(shortcut + self.dropout(src))

        shortcut = src
        src = self.feedforward(src)
        src = self.feedforward_ln(shortcut + self.dropout(src))

        return src


class Encoder(nn.Module):
    def __init__(self, entity_input_dim, relation_input_dim, hid_dim, n_layers, n_heads,
                 pf_dim, dropout, max_length=256):
        super().__init__()

        self.entity_embedding = nn.Embedding(entity_input_dim, hid_dim // 2)
        self.relation_embedding = nn.Embedding(relation_input_dim, hid_dim // 2)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([
            EncoderLayer(
                hid_dim, n_heads, pf_dim, dropout
            )
            for _ in range(n_layers)
        ])

        self.scale = np.sqrt(hid_dim)
        self.dropout = nn.Dropout(dropout)

        # TUPE
        # self.fc_pos_q = nn.Linear(hid_dim, hid_dim)
        # self.fc_pos_k = nn.Linear(hid_dim, hid_dim)
        # self.n_heads = n_heads
        # self.head_dim = hid_dim // n_heads

    def forward(self, entities, relations, mask):
        device = next(self.parameters()).device

        batch_size, trajectory_len = entities.shape

        pos = torch.arange(trajectory_len).repeat(batch_size, 1).to(device)

        # TUPE
        # pos_src = self.pos_embedding(pos)
        # pos_Q = self.fc_pos_q(pos_src).view(batch_size, -1, self.n_heads, self.head_dim)
        # pos_K = self.fc_pos_k(pos_src).view(batch_size, -1, self.n_heads, self.head_dim)
        # abs_pos_energy = torch.einsum('bqnh,bknh->bnqk', pos_Q, pos_K)

        try:
            entities = self.entity_embedding(entities)
        except:
            print(entities)
        relations = self.relation_embedding(relations)

        embeddings = torch.cat((entities, relations), -1)

        x = self.dropout(embeddings * self.scale)

        for layer in self.layers:
            x = layer(x, mask)

        return x


class Transformer(nn.Module):
    def __init__(self, encoder, entity_input_dim, relation_input_dim,
                 hid_dim, output_dim, entity_pad_idx, relation_pad_idx):
        super().__init__()

        self.encoder = encoder
        self.entity_pad_idx = entity_pad_idx
        self.relation_pad_idx = relation_pad_idx

        if config['only_relations']:
            self.relation_embedding = nn.Embedding(relation_input_dim, hid_dim)
        else:
            self.entity_embedding = nn.Embedding(entity_input_dim, hid_dim // 2)
            self.relation_embedding = nn.Embedding(relation_input_dim, hid_dim // 2)

        # self.fc = nn.Linear(hid_dim, output_dim)

        # if config['rl_method'] == 'A2C':
        #     self.fc_v = nn.Linear(hid_dim, 1)

    def make_mask(self, entities):
        entities_len = entities.shape[1]
        mask = (entities != self.entity_pad_idx).view(-1, 1, 1, entities_len)
        return mask

    def forward(self, entities, relations, actions, action_keys):
        mask = self.make_mask(entities)
        encoded = self.encoder(entities, relations, mask)

        # get outputs from the last token for each sequence
        idxs = mask.sum([1, 2, 3]) - 1
        idxs = idxs.view(-1, 1, 1).repeat(1, 1, encoded.shape[-1])
        encoded = encoded.gather(1, idxs).squeeze(1)

        if config['only_relations']:
            action_embedding = self.relation_embedding(actions)
        else:
            r, e = actions
            relation_embedding = self.relation_embedding(r)
            entity_embedding = self.entity_embedding(e)
            action_embedding = torch.cat([relation_embedding, entity_embedding], dim=-1)

        # if config['rl_method'] == 'A2C':
        #     return self.fc(encoded), self.fc_v(encoded)

        preds = []
        for i, keys in enumerate(action_keys):
            preds.append(action_embedding[keys] @ encoded[i])

        return preds

        # return self.fc(encoded)


class GraphSearchPolicy(nn.Module):
    def __init__(self, entity_input_dim, relation_input_dim, entity_dim=200, relation_dim=200, history_dim=200):
        super().__init__()

        if config['only_relations']:
            action_dim = relation_dim
            self.relation_embedding = nn.Embedding(relation_input_dim, relation_dim)
        else:
            action_dim = relation_dim + entity_dim
            self.entity_embedding = nn.Embedding(entity_input_dim, entity_dim)
            self.relation_embedding = nn.Embedding(relation_input_dim, relation_dim)

        input_dim = history_dim + entity_dim + relation_dim
        self.path_encoder = nn.LSTM(
            input_size=action_dim,
            hidden_size=history_dim,
            num_layers=3,
            batch_first=True
        )

        self.W1 = nn.Linear(input_dim, action_dim)
        self.W2 = nn.Linear(action_dim, action_dim)
        self.relu = nn.ReLU()
        self.dp = nn.Dropout(0.1)

        # self.initialize_modules()

    def forward(self, path, e, q, actions, action_keys):        
        Q = self.relation_embedding(q)
        H = path[-1][0][-1, :, :]

        E = self.entity_embedding(e)
        X = torch.cat([E, H, Q], dim=-1)

        X = self.dp(self.relu(self.W1(X)))
        X2 = self.dp(self.W2(X))

        if config['only_relations']:
            action_embedding = self.relation_embedding(actions)
        else:
            rs, es = actions
            relation_embedding = self.relation_embedding(rs)
            entity_embedding = self.entity_embedding(es)
            action_embedding = torch.cat([relation_embedding, entity_embedding], dim=-1)

        # output = torch.squeeze(action_embedding @ torch.unsqueeze(X2, 2), 2)

        preds = []
        for i, keys in enumerate(action_keys):
            # print(X2[i])
            preds.append(action_embedding[keys] @ X2[i])
        return preds

    def initialize_path(self, init_actions):
        device = next(self.parameters()).device

        # [batch_size, action_dim]
        if config['only_relations']:
            init_action_embedding = self.relation_embedding(init_actions)
        else:
            rs, es = init_actions
            relation_embedding = self.relation_embedding(rs)
            entity_embedding = self.entity_embedding(es)
            init_action_embedding = torch.cat([relation_embedding, entity_embedding], dim=-1)
    
        init_action_embedding.unsqueeze_(1)
        # [num_layers, batch_size, dim]
        init_h = torch.zeros([3, len(init_action_embedding), 200], device=device)
        init_c = torch.zeros([3, len(init_action_embedding), 200], device=device)
        return [list(self.path_encoder(init_action_embedding, (init_h, init_c))[1])]

    def update_path(self, path, actions):
        device = next(self.parameters()).device
        
        if config['only_relations']:
            action_embedding = self.relation_embedding(actions)
        else:
            rs, es = actions
            relation_embedding = self.relation_embedding(torch.tensor(rs, device=device))
            entity_embedding = self.entity_embedding(torch.tensor(es, device=device))
            action_embedding = torch.cat([relation_embedding, entity_embedding], dim=-1)

        return path + [list(self.path_encoder(action_embedding.unsqueeze(1), path[-1])[1])]

    def initialize_modules(self):
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)
        for name, param in self.path_encoder.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)