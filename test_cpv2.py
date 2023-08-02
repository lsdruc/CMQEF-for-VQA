import argparse
import torch
from torch.utils.data import DataLoader
from dataset_cpv2 import Dictionary, VQAFeatureDataset
from train_cpv2 import evaluate
import os
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

if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    batch_size = args.batch_size
    eval_dset = VQAFeatureDataset('val', dictionary)
    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=0, pin_memory=False)
    mymn=os.listdir('saved_models/CMQEF')
    for i in mymn:
        if i not in 'cpv2train.log':
            model = torch.load('saved_models/CMQEF/' + i).cuda()
            model.eval()
            evaluate(model, eval_loader, args.output, i)
