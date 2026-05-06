import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from prettytable import PrettyTable
from tqdm import tqdm
from dataset import HybridFieldChannelDataset


def model_params(net):
    table = PrettyTable(["Network Component", "# Parameters"])
    num_params = 0
    for name, parameter in net.named_parameters():
        if not parameter.requires_grad:
            continue
        table.add_row([name, parameter.numel()])
        num_params += parameter.numel()
    table.add_row(['TOTAL', num_params])
    return table


class NMSELoss(nn.Module):
    def __init__(self):
        super(NMSELoss, self).__init__()

    def forward(self, h_ideal, h_predict, reduction='mean'):
        nmse = (torch.norm(h_predict-h_ideal, p=2, dim=-1)**2 / torch.norm(h_ideal, p=2, dim=-1)**2)
        if reduction == None:
            return nmse
        elif reduction == 'mean':
            return nmse.mean()
        elif reduction == 'mean batch':
            return nmse.mean(0)


def train(model, n_epochs, scheduler, train_loader, test_loader, optimizer, loss_fn, loss_method, saver, writer):
    best_train_nmse = np.inf
    best_test_nmse = np.inf
    print(model)
    print(model_params(model))

    for epoch in range(n_epochs):
        model.train()
        avg_train_nmse = 0.0
        avg_test_nmse = 0.0
        pbar = tqdm(train_loader, leave=False)
        train_loader.dataset.update_dataset(epoch)                  # regenerate the dataset. pass the epoch idx as the initial seed
        for channels, measurements in pbar:
            train_nmse = 0
            optimizer.zero_grad()
            pred_channels_list = model(measurements)
            pred_channels = pred_channels_list[-1]
            train_nmse += loss_fn(channels, pred_channels)
            train_nmse.backward()
            optimizer.step()
            avg_train_nmse += train_nmse.item()
            pbar.set_description(f"epoch {epoch+1:4d}/{n_epochs}]: train NMSE {10*np.log10(train_nmse.item()):.3f}")
        avg_train_nmse /= len(train_loader)
        scheduler.step(avg_train_nmse)
        train_loader.dataset.delete_dataset()

        pbar = tqdm(test_loader, leave=False)
        model.eval()
        test_loader.dataset.update_dataset(n_epochs+epoch)        # regenerate the dataset. pass the epoch idx as the initial seed
        with torch.no_grad():
            for channels, measurements in pbar:
                pred_channels_list = model(measurements)
                pred_channels = pred_channels_list[-1]
                test_nmse = loss_fn(channels, pred_channels)
                avg_test_nmse += test_nmse.item()
                pbar.set_description(f"epoch {epoch+1:4d}/{n_epochs}]: test NMSE {10*np.log10(test_nmse.item()):.3f}")
            avg_test_nmse /= len(test_loader)
        test_loader.dataset.delete_dataset()

        if best_train_nmse > avg_train_nmse:
            best_train_nmse = avg_train_nmse
        if best_test_nmse > avg_test_nmse:
            best_test_nmse = avg_test_nmse
        
        if writer == True:
            wandb.log({
                'Train NMSE': avg_train_nmse,
                'Test NMSE': avg_test_nmse
                      })

        print(f"epoch {epoch+1:4d}/{n_epochs:4d}]: LR {scheduler.get_last_lr()[0]:.3e}, train NMSE {10*np.log10(avg_train_nmse):.3f}, test NMSE {10*np.log10(avg_test_nmse):.3f}, best train NMSE {10*np.log10(best_train_nmse):.3f}, best test NMSE {10*np.log10(best_test_nmse):.3f}")
        saver(avg_test_nmse, epoch, model)
    
    if writer == True:
        wandb.finish()


def evaluate(model, loss_fn, **kwargs):
    snrs = np.arange(kwargs.get('snr_db')['start'], kwargs.get('snr_db')['stop']+kwargs.get('snr_db')['step'], kwargs.get('snr_db')['step'])
    nmse_list = []

    model.eval()
    with torch.no_grad():
        pbar_outer = tqdm(snrs, desc='', leave=False)
        for snr_idx, snr in enumerate(pbar_outer):
            dataset = HybridFieldChannelDataset(**kwargs.get('dataset'), snr_db_min=snr, snr_db_max=snr, device=model.device())     # TODO: parse dataset device from config
            dataset.update_dataset(0xDEADBEEF)
            loader = DataLoader(dataset=dataset, **kwargs.get('loader'))
            avg_nmse = np.zeros(dataset.n_sc)
            pbar_inner = tqdm(loader, leave=False)
            for channels, measurements in pbar_inner:
                pred_channels_list = model(measurements)
                nmse = loss_fn(channels, pred_channels_list[-1], reduction='mean batch')
                avg_nmse += nmse.cpu().numpy()
                pbar_inner.set_description(f"SNR {snr:.3f} dB: NMSE {10*np.log10(nmse.mean().item()):.3f} dB")
            dataset.delete_dataset()
            avg_nmse /= len(loader)
            nmse_list.append(avg_nmse)
            data = np.concatenate((snrs[:snr_idx+1,None],10*np.log10(np.array(nmse_list).mean(1))[:,None],10*np.log10(np.array(nmse_list))), axis=1)
            np.savetxt(kwargs.get('results_dir')+kwargs.get('load_checkpoint_name')+kwargs.get('comment')+'.txt', data, header='SNR NMSE')


class SaveBestModel:
    def __init__(self, path, verbose, save_every=None, best_metric=float('inf'), **kwargs):
        self.best_metric = best_metric
        self.path = path
        self.verbose = verbose
        self.save_every = save_every

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

    def __call__(self, current_metric, epoch, model):
        if current_metric < self.best_metric:
            self.best_metric = current_metric
            if self.verbose:
                print(f"\nBest metric: {self.best_metric}")
                print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict()
                }, self.path)
        if self.save_every:
            if not (epoch % self.save_every):
                sp_path = self.path.split('.')
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict()
                }, f'{sp_path[0]}_ep_{epoch}.{sp_path[-1]}')

    @staticmethod
    def save_at_exit(model, path):
        sp_path = path.split('.')
        torch.save({'model_state_dict': model.state_dict()}, f'{sp_path[0]}_last_ep.{sp_path[-1]}')
