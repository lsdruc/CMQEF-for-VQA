import os
import time
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
from tqdm import tqdm
import logging
def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)

    loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


#def train(model, train_loader, eval_loader, num_epochs, output):
def train(model, train_loader,num_epochs, output):
    utils.create_dir(output)
    optim = torch.optim.Adamax(model.parameters())
    logger = utils.Logger(os.path.join(output, 'train.log'))
    best_eval_score = 0

    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        t = time.time()

        for i, (v, b, q, a) in enumerate(train_loader):
            v = Variable(v).cuda()
            b = Variable(b).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()

            pred = model(v, b, q, a)
            loss = instance_bce_with_logits(pred, a)
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            #nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optim.step()
            #optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data).sum()
            total_loss += loss.data[0] * v.size(0)
            #total_loss += loss.item() * v.size(0)
            train_score += batch_score

        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)
        logger.write('epoch %d, time: %.2f' % (epoch+1, time.time()-t))
        logger.write('\ttrain_loss: %.4f, score: %.4f' % (total_loss, train_score))
        logger.write('%d,%.2f,%.6f,%.4f,%.4f' % (epoch + 1, time.time() - t, optim.param_groups[0]['lr'], total_loss, train_score))
        #logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))
        file_name ='vqa_epoch'+'%d'%(epoch+1)+'_' + '%.4f' % (train_score) + '_model.pth'
        model_path = os.path.join(output, file_name)#xin jia
        #torch.save(model.state_dict(), model_path)#xin jia
        torch.save(model, model_path)


def evaluate(model, dataloader,output,model_name):
    logging.basicConfig(level=logging.DEBUG, filename='test.log')
    logger = logging.getLogger(output)
    #utils.create_dir(output)
    #logger = utils.Logger(os.path.join(output, 'testlog.txt'))
    score = 0
    V_loss=0
    total_loss = 0
    upper_bound = 0
    val_half_upper_bound=0
    test_half_upper_bound = 0
    num_data = 0
    #score_val_half=0
    val_half_loss=0
    val_half_score=0
    val_half_num=len(dataloader.dataset)/2
    test_half_num=len(dataloader.dataset)-val_half_num

    test_half_loss=0
    test_half_score=0
    #for v, b, q, a in iter(dataloader):
    for v, b, q, a in tqdm(dataloader, ncols=100, total=len(dataloader), desc="eval"):
        v = Variable(v, volatile=True).cuda()
        b = Variable(b, volatile=True).cuda()
        q = Variable(q, volatile=True).cuda()
        a = Variable(a, volatile=True).cuda()
        pred = model(v, b, q, None)
        loss = instance_bce_with_logits(pred, a)
        V_loss += loss.data[0] * v.size(0)
        #loss = instance_bce_with_logits(pred, a)
        #loss=0
        batch_score = compute_score_with_logits(pred, a.data).sum()
        score += batch_score
        #total_loss += loss.data[0] * v.size(0)
        #upper_bound += (a.max(1)[0]).sum()#yuanlai
        upper_bound += (a.data.max(1)[0]).sum()
        num_data += pred.size(0)
        if num_data<val_half_num:
            val_half_loss=V_loss
            val_half_score=score
            val_half_upper_bound=upper_bound
        else:
            test_half_loss=V_loss-val_half_loss
            test_half_score=score-val_half_score
            test_half_upper_bound = upper_bound-val_half_upper_bound

    #total_loss /= len(dataloader.dataset)
    V_loss = 100*V_loss/len(dataloader.dataset)
    score = 100*score / len(dataloader.dataset)
    upper_bound = upper_bound*100 / len(dataloader.dataset)

    val_half_loss=100*val_half_loss/val_half_num
    val_half_score = 100*val_half_score / val_half_num
    val_half_upper_bound = val_half_upper_bound*100 / val_half_num

    test_half_loss = 100*test_half_loss/test_half_num
    test_half_score = 100*test_half_score / test_half_num
    test_half_upper_bound = test_half_upper_bound*100 / test_half_num

    #logger.write('The result of model: %s for val_all, val_half and test_half' % model_name)
    #logger.write('\tval_all_score: %.4f, val_all_upper_bound:%.4f' % (score,upper_bound))
    #logger.write('\tval_half_score: %.4f, test_all_upper_bound:%.4f' % (val_half_score,val_half_upper_bound))
    #logger.write('\ttest_half_score: %.4f,, test_all_upper_bound:%.4f' % (test_half_score, test_half_upper_bound))

    logger.info('The result of model: %s for val_all, val_half and test_half' % model_name)
    logger.info('\tval_all_loss:%.4f, val_all_score: %.4f, val_all_upper_bound:%.4f' % (V_loss,score,upper_bound))
    logger.info('\tval_half_score:%.4f, val_half_score: %.4f, val_half_upper_bound:%.4f' % (val_half_score, val_half_score,val_half_upper_bound))
    logger.info('\ttest_half_loss:%.4f, test_half_score: %.4f, test_half_upper_bound:%.4f' % (test_half_loss,test_half_score, test_half_upper_bound))
    logger.info('%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f'%(model_name,V_loss,score,val_half_loss,val_half_score,test_half_loss,test_half_score))

    #return score, upper_bound