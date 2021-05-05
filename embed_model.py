import torch
from torch import nn
import torch.nn.functional as F


class ComplEx(nn.Module):
    def __init__(self, entities_num, relations_num, hid_dim=256, dp=0.2):
        super().__init__()

        self.entity_re_emb = nn.Embedding(entities_num, hid_dim)
        self.entity_im_emb = nn.Embedding(entities_num, hid_dim)
        self.relation_re_emb = nn.Embedding(relations_num, hid_dim)
        self.relation_im_emb = nn.Embedding(relations_num, hid_dim)

        self.dp = nn.Dropout(dp)
        self.sigmoid = nn.Sigmoid()


    def forward(self, e1, r, e2=None):
        def dist_mult(E1, R, E2):
            return torch.mm(E1 * R, E2.transpose(1, 0))

        if e2 is None:
            e2_re = self.dp(self.entity_re_emb.weight)
            e2_im = self.dp(self.entity_im_emb.weight)
        else:
            e2_re = self.dp(self.entity_re_emb(e2))
            e2_im = self.dp(self.entity_im_emb(e2))

        e1_re = self.dp(self.entity_re_emb(e1))
        e1_im = self.dp(self.entity_im_emb(e1))
        r_re = self.dp(self.relation_re_emb(r))
        r_im = self.dp(self.relation_im_emb(r))

        rrr = dist_mult(r_re, e1_re, e2_re)
        rii = dist_mult(r_re, e1_im, e2_im)
        iri = dist_mult(r_im, e1_re, e2_im)
        iir = dist_mult(r_im, e1_im, e2_re)
        S = rrr + rii + iri - iir
        return S

    # def forward_fact(self, e1, r, e2):
    #     def dist_mult(E1, R, E2):
    #         return torch.mm(E1 * R, E2.transpose(1, 0))

    #     # if e2 is None:
    #     #     e2_re = self.dp(self.entity_re_emb.weight)
    #     #     e2_im = self.dp(self.entity_im_emb.weight)
    #     # else:
    #     e2_re = self.dp(self.entity_re_emb(e2))
    #     e2_im = self.dp(self.entity_im_emb(e2))

    #     e1_re = self.dp(self.entity_re_emb(e1))
    #     e1_im = self.dp(self.entity_im_emb(e1))
    #     r_re = self.dp(self.relation_re_emb(r))
    #     r_im = self.dp(self.relation_im_emb(r))

    #     rrr = dist_mult(r_re, e1_re, e2_re)
    #     rii = dist_mult(r_re, e1_im, e2_im)
    #     iri = dist_mult(r_im, e1_re, e2_im)
    #     iir = dist_mult(r_im, e1_im, e2_re)
    #     S = rrr + rii + iri - iir
    #     S = self.sigmoid(S)
    #     return S
