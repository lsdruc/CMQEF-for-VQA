import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_cpv2 import Dictionary, VQAFeatureDataset
import base_model
from train_cpv2 import train
import utils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='CMQEF_UpDn')
    parser.add_argument('--output', type=str, default='saved_models/CMQEF')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    args = parser.parse_args()
    return args



def weights_init_kn(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight.data, a=0.25)
def weights_init_ku(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform(m.weight.data, a=0.25)


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    train_dset = VQAFeatureDataset('train', dictionary)
    #eval_dset = VQAFeatureDataset('val', dictionary)
    #train_dset = VQAFeatureDataset('val', dictionary)
    batch_size = args.batch_size

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(train_dset, args.num_hid).cuda()
    model.apply(weights_init_kn)
    model.w_emb.init_embedding('data/glove6b_init_300d.npy')
    model = model.cuda()

    #if torch.cuda.device_count()>1:
        #model = nn.DataParallel(model).cuda()
    #else:
        #model =model.cuda()

    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=0)
    #eval_loader =  DataLoader(eval_dset, batch_size, shuffle=False, num_workers=0)
    #train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=1,pin_memory=True)
    #eval_loader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=1,pin_memory=True)
    #train(model, train_loader, eval_loader, args.epochs, args.output)
    train(model, train_loader, args.epochs, args.output)
    #train(model, train_loader, eval_loader, args.epochs, args.output)