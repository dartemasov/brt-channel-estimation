import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import atexit
import argparse
import json
import time
import wandb
from models.block_recurrent_transformer_pytorch import BlockRecurrentTransformer
from utils import NMSELoss, SaveBestModel, train, evaluate
from dataset import HybridFieldChannelDataset

if __name__ == '__main__':

    default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    time_str = time.strftime("_%m%d_%H%M", time.localtime())

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='both', type=str, choices=['train', 'eval', 'both'])
    parser.add_argument('--config', default='configs/default.json', type=str, help='configuration filepath')
    parser.add_argument('--device', default=default_device, type=str, help='device for computations (cuda:#/cpu)')

    args = parser.parse_args()

    with open(args.config, 'rt') as f:
        args.__dict__.update(json.load(f))
    print('device:', args.device)

    model = getattr(sys.modules[__name__], args.model.pop('name'))(args.device, args.training['dataset']['meas_mat_path'], **args.model).to(args.device)
    loss_fn = NMSELoss()

    if args.mode == 'train' or args.mode == 'both':

        if args.training['load_checkpoint_name']:
            pth = torch.load(args.training['checkpoints_dir']+args.training['load_checkpoint_name']+'.pth', map_location=args.device, weights_only=True)
            model.load_state_dict(pth['model_state_dict'])
            print('Checkpoint loaded successfully')

        if args.training['enable_writer']:
            wandb.init(
                project="nfc-chest",
                config=vars(args),
                name=args.training['comment']
            )

        optimizer = optim.AdamW(model.parameters(), lr=args.training['learning_rate'], weight_decay=args.training['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=25, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-7, eps=1e-08)

        train_dataset = HybridFieldChannelDataset(**args.training['dataset'], device=args.device)
        test_dataset = HybridFieldChannelDataset(**args.validation['dataset'], device=args.device)

        train_loader = DataLoader(dataset=train_dataset, **args.training['loader'])
        test_loader = DataLoader(dataset=test_dataset, **args.validation['loader'])

        save_checkpoint_path = args.training['checkpoints_dir'] + args.training['save_checkpoint_name'] + '.pth'
        saver = SaveBestModel(path=save_checkpoint_path, verbose=True, save_every=None)
        atexit.register(saver.save_at_exit, model, save_checkpoint_path)                        # save last epoch before termination

        # start training
        train(model, args.training['n_epochs'], scheduler, train_loader, test_loader, optimizer, loss_fn, args.training['loss_method'], saver, args.training['enable_writer'])
        saver.save_at_exit(model, save_checkpoint_path)
        atexit.unregister(saver.save_at_exit)

    if args.mode != 'train':
        if args.mode == 'eval':
            pth = torch.load(args.evaluation['checkpoints_dir']+args.evaluation['load_checkpoint_name']+'.pth', map_location=args.device, weights_only=True)
            print(f"Model checkpoint loaded from {args.evaluation['checkpoints_dir']+args.evaluation['load_checkpoint_name']+'.pth'}")
        elif args.mode == 'both':
            pth = torch.load(args.training['checkpoints_dir']+args.training['save_checkpoint_name']+'.pth', map_location=args.device, weights_only=True)
            print(f"Model checkpoint loaded from {args.training['checkpoints_dir']+args.training['save_checkpoint_name']+'.pth'}")
        model.load_state_dict(pth['model_state_dict'])
        args.evaluation['comment'] += time_str

        evaluate(model, loss_fn, **args.evaluation)

        if args.evaluation['eval_last_epoch']:
            if args.mode == 'eval':
                pth = torch.load(args.evaluation['checkpoints_dir']+args.evaluation['load_checkpoint_name']+'_last_ep.pth', map_location=args.device, weights_only=True)
                print(f"Model checkpoint loaded from {args.evaluation['checkpoints_dir']+args.evaluation['load_checkpoint_name']+'_last_ep.pth'}")
            elif args.mode == 'both':
                pth = torch.load(args.training['checkpoints_dir']+args.training['save_checkpoint_name']+'_last_ep.pth', map_location=args.device, weights_only=True)
                print(f"Model checkpoint loaded from {args.training['checkpoints_dir']+args.training['save_checkpoint_name']+'_last_ep.pth'}")
            model.load_state_dict(pth['model_state_dict'])
            args.evaluation['comment'] += '_last_ep'

            evaluate(model, loss_fn, **args.evaluation)
