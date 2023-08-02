import torch
from sklearn.metrics.pairwise import pairwise_distances
import torch.nn as nn
from attention import Att_2
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet

class BaseModel(nn.Module):

    def __init__(self, w_emb, q_emb, v_att,q_net, v_net, classifier):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier

    def forward(self, v, b, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        #v:(256L, 36L, 2048L)
        #q:(256L, 14L)
        #w_emb:(256L, 14L, 300L)
        #q_emb:(256L, 1024L)
        #att:(256L, 36L, 1L)
        #v_emb:(256L, 2048L)
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        #for i in range(256):
            #for j in range(1024):
                #score[i][j]=cosine_similarity(torch.FloatTensor(q_repr[i][j]),torch.FloatTensor(v_repr[i][j]),dim=0,eps=1e-8)
        #joint_repr = q_repr * v_repr #yuanlai
        joint_repr = q_repr * v_repr  # yuanlai




        #q_emb3 = q_repr1 - v_repr1
        #joint_repr2=torch.cat((q_repr,v_repr), 1)
        #vq2 = self.v_net(joint_repr2)

        #joint_repr = q_emb2+q_emb3
        #joint_repr1 = q_emb2 * v_repr
        #joint_repr2 = joint_repr1 * q_emb1
        #joint_repr = joint_repr2 * v_repr1

        #joint_repr = pairwise_distances(q_repr, v_repr, metric="euclidean")
        #joint_repr = q_repr + v_repr#xin zeng
        #joint_repr = q_repr - v_repr  # xin zeng
        #joint_repr=torch.cat((q_repr, v_repr), 2)

        logits = self.classifier(joint_repr)

        return logits



def build_CMQEF_UpDn(dataset, num_hid):
    #newatt
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, dropout=0.35)
    q_emb = QuestionEmbedding(300, num_hid, 1, True, dropout=0.5)
    v_att = Att_2(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att,q_net, v_net, classifier)

