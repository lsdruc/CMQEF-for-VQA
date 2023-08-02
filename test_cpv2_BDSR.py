import argparse
import torch
from torch.utils.data import DataLoader
from dataset_cpv2 import Dictionary, VQAFeatureDataset
from train_cpv2_BDSR import evaluate
import os
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=60)
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
    #train_dset = VQAFeatureDataset('train', dictionary)
    batch_size = args.batch_size
    eval_dset = VQAFeatureDataset('val', dictionary)
    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=0, pin_memory=False)
    #constructor = 'build_%s' % args.model
    #model = getattr(base_model, constructor)(train_dset, args.num_hid).cuda()
    #model.w_emb.init_embedding('data/glove6b_init_300d.npy')
    mymn=os.listdir('saved_models/CMQEF')
    for i in mymn:
        if i not in 'cpv2trainBDSR.log':
            model = torch.load('saved_models/CMQEF/' + i).cuda()
            model.eval()
            evaluate(model, eval_loader, args.output, i)

    #model_name1 = 'vqa_epoch2_52.9227_model1.pth'
    #model_load=torch.load(open('saved_models/exp0/' + model_name1))
    #model = torch.load('saved_models/exp0/' + model_name1).cuda()
    #model.eval()
    #if torch.cuda.device_count()>1:
        #model = nn.DataParallel(model).cuda()
    #else:
        #model=model.cuda()
    #model.load_state_dict(model_load)

    #del train_dset
    #gc.collect()
    #eval_dset = VQAFeatureDataset('val', dictionary)

    #train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=0,pin_memory=False)
    #eval_loader =  DataLoader(eval_dset, batch_size, shuffle=False, num_workers=0,pin_memory=False)
    #train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=1,pin_memory=True)
    #eval_loader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=1,pin_memory=True)

    #train(model, train_loader, eval_loader, args.epochs, args.output)


    #evaluate(model, eval_loader,args.output, model_name1)