import time
from random import randrange

import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import *

def train_model(args, model, dataloader, criterion, optimizer, lr_scheduler, writer, device='cpu', val_dataloader=None, start_time=None): 
    history = {
        'training_time(min)': None
    }
    
    # if args.loss_type.lower() == 'geco':
    #     history.update( {
    #         'lambda': []
    #     } )

    if val_dataloader is not None:
        # history.update( {
        #     'mean_val_loss_total': [],
        #     'mean_val_loss_reconstruction': [],
        #     'mean_val_loss_kl_term': [],
        #     'mean_val_loss_kl': [[] for _ in range(args.latent_num)]
        # } )

        val_minibatches = len(val_dataloader)


    def record_history(idx, loss_dict, type='train'):
        prefix = 'Minibatch Training ' if type == 'train' else 'Mean Validation '

        loss_per_pixel = loss_dict['loss'].item() / args.pixels
        reconstruction_per_pixel = loss_dict['reconstruction_term'].item() / args.pixels
        kl_term_per_pixel = loss_dict['kl_term'].item() / args.pixels
        kl_per_pixel = [ loss_dict['kls'][v].item() / args.pixels for v in range(args.latent_num) ]

        # Total Loss
        _dict = {   'total': loss_per_pixel,
                    'kl term': kl_term_per_pixel, 
                    'reconstruction': reconstruction_per_pixel  }
        writer.add_scalars(prefix + 'Loss Curve', _dict, idx)
        
        # Reconstruction Term Decomposition
        if args.rec_type.lower() == 'loglikelihood':
            _dict = {   'reconstruction term mse': loss_dict['recterm_mse'].item() / args.pixels,
                        'reconstruction term std': loss_dict['recterm_std'].item() / args.pixels   }
            writer.add_scalars(prefix + 'Loss Curve (Reconstruction)', _dict, idx)

        # KL Term Decomposition
        _dict = { 'sum': sum(kl_per_pixel) }
        _dict.update( { 'scale {}'.format(v): kl_per_pixel[v] for v in range(args.latent_num) } )
        writer.add_scalars(prefix + 'Loss Curve (K-L)', _dict, idx)

        # Coefficients
        if type == 'train':
            if args.loss_type.lower() == 'elbo':
                writer.add_scalar('Beta', criterion.beta_scheduler.beta, idx)
            elif args.loss_type.lower() == 'geco':
                writer.add_scalar('Lagrange Multiplier', criterion.log_inv_function(criterion.log_lamda).item(), idx)


    val_images, val_truths = next(iter(val_dataloader))
    val_images, val_truths = val_images[:16], val_truths[:16]
    truth_grid = make_grid(val_truths, nrow=4, pad_value=val_truths.min().item())
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(truth_grid[0])
    ax.set_axis_off()
    fig.tight_layout()
    writer.add_figure('Validation Images / Ground Truth', fig)
    val_images_selection = val_images.to(device)
    
    last_time_checkpoint = start_time
    for e in range(args.epochs):
        for mb, (images, truths) in enumerate(tqdm(dataloader)):
            idx = e*len(dataloader) + mb+1

            # Initialization
            criterion.train()
            model.train()
            model.zero_grad()
            images, truths = images.to(device), truths.to(device)

            # Train One Step
            
            ## Get Predictions and Prepare for Loss Calculation
            if args.rec_type.lower() == 'mse':
                preds, infodicts = model(images, truths)
                preds, infodict = preds[:,0], infodicts[0]
                logstd2 = None

            elif args.rec_type.lower() == 'loglikelihood':
                if args.ll_std_sample_num == 0:
                    preds, infodicts = model(images, truths, first_channel_only=False)
                    preds, logstd2, infodict = preds[:,0,0], preds[:,0,1], infodicts[0]

                elif args.ll_std_sample_num > 0:
                    preds, infodicts = model(images, truths, times=args.ll_std_sample_num)
                    # with torch.no_grad():
                    logstd2 = torch.log( torch.var(preds, dim=1, unbiased=True) )  # or preds.detach()
                    k = randrange(args.ll_std_sample_num)
                    preds, infodict = preds[:,k], infodicts[k]

            truths = truths.squeeze(dim=1)


            ## Calculate Loss
            loss = criterion(preds, truths, kls=infodict['kls'], logstd2=logstd2, lr=lr_scheduler.get_last_lr()[0])


            ## Backpropagate
            loss.backward()             # Calculate Gradients
            optimizer.step()            # Update Weights
            

            ## Step Beta Scheduler
            if args.loss_type.lower() == 'elbo':
                criterion.beta_scheduler.step()


            # Record Train History
            loss_dict = criterion.last_loss.copy()
            loss_dict.update( { 'kls': infodict['kls'] } )

            if args.rec_type.lower() == 'loglikelihood':
                if isinstance(criterion.reconstruction_loss, TopkMaskedLoss):
                    ll_loss_internal = criterion.reconstruction_loss.loss.last_loss
                else:
                    ll_loss_internal = criterion.reconstruction_loss.last_loss
                
                loss_dict.update( { 'recterm_mse': ll_loss_internal['expanded_mse_term'].sum(dim=(1,2)).mean(),
                                    'recterm_std': ll_loss_internal['expanded_std_term'].sum(dim=(1,2)).mean() } )

            record_history(idx, loss_dict)
            
            
            # Validation
            if idx % args.val_period == 0 and val_dataloader is not None:
                criterion.eval()
                model.eval()

                # Show Sample Validation Images
                with torch.no_grad():
                    val_preds = model(val_images_selection)[0]
                    
                    out_grid = make_grid(val_preds, nrow=4, pad_value=val_preds.min().item())

                    fig, ax = plt.subplots(figsize=(6,6))
                    ax.imshow(out_grid[0].cpu())
                    ax.set_axis_off()
                    fig.tight_layout()
                    writer.add_figure('Validation Images / Prediction', fig, idx)

                # Calculate Validation Loss
                mean_val_loss, mean_val_reconstruction_term, mean_val_kl_term = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
                mean_val_kl = torch.zeros(args.latent_num, device=device)
                mean_val_recterm_mse, mean_val_recterm_std = torch.zeros(1, device=device), torch.zeros(1, device=device)

                with torch.no_grad():
                    for _, (val_images, val_truths) in enumerate(val_dataloader):
                        val_images, val_truths = val_images.to(device), val_truths.to(device)
                        
                        ## Get Predictions and Prepare for Loss Calculation
                        if args.rec_type.lower() == 'mse':
                            preds, infodicts = model(val_images, val_truths)
                            preds, infodict = preds[:,0], infodicts[0]
                            logstd2 = None

                        elif args.rec_type.lower() == 'loglikelihood':
                            if args.ll_std_sample_num == 0:
                                preds, infodicts = model(val_images, val_truths, first_channel_only=False)
                                preds, logstd2, infodict = preds[:,0,0], preds[:,0,1], infodicts[0]

                            elif args.ll_std_sample_num > 0:
                                preds, infodicts = model(val_images, val_truths, times=args.ll_std_sample_num)
                                logstd2 = torch.log( torch.var(preds.detach(), dim=1, unbiased=True) )
                                k = randrange(args.ll_std_sample_num)
                                preds, infodict = preds[:,k], infodicts[k]

                        val_truths = val_truths.squeeze(dim=1)

                        ## Calculate Loss
                        loss = criterion(preds, val_truths, kls=infodict['kls'], logstd2=logstd2)


                        mean_val_loss += loss
                        mean_val_reconstruction_term += criterion.last_loss['reconstruction_term']
                        mean_val_kl_term += criterion.last_loss['kl_term']
                        mean_val_kl += infodict['kls']
                        
                        if args.rec_type.lower() == 'loglikelihood':
                            if isinstance(criterion.reconstruction_loss, TopkMaskedLoss):
                                ll_loss_internal = criterion.reconstruction_loss.loss.last_loss
                            else:
                                ll_loss_internal = criterion.reconstruction_loss.last_loss
                            
                            mean_val_recterm_mse += ll_loss_internal['expanded_mse_term'].sum(dim=(1,2)).mean()
                            mean_val_recterm_std += ll_loss_internal['expanded_std_term'].sum(dim=(1,2)).mean()
                    

                    mean_val_loss /= val_minibatches
                    mean_val_reconstruction_term /= val_minibatches
                    mean_val_kl_term /= val_minibatches
                    mean_val_kl /= val_minibatches
                    mean_val_recterm_mse /= val_minibatches
                    mean_val_recterm_std /= val_minibatches


                # Record Validation History
                loss_dict = {
                    'loss': mean_val_loss,
                    'reconstruction_term': mean_val_reconstruction_term,
                    'kl_term': mean_val_kl_term,
                    'kls': mean_val_kl,
                    'recterm_mse': mean_val_recterm_mse,
                    'recterm_std': mean_val_recterm_std
                }
                record_history(idx, loss_dict, type='val')
        
        
        # Report Epoch Completion
        time_checkpoint = time.time()
        epoch_time = (time_checkpoint - last_time_checkpoint) / 60
        total_time = (time_checkpoint - start_time) / 60
        print('Epoch {}/{} done in {:.1f} minutes. \t\t\t\t Total time: {:.1f} minutes'.format(e+1, args.epochs, epoch_time, total_time))
        last_time_checkpoint = time_checkpoint
        
        # Save Model and Loss
        if (e+1) % args.save_period == 0 and (e+1) != args.epochs:
            torch.save(model, '{}/{}/model{}.pth'.format(args.output_dir, args.stamp, e+1))
            torch.save(criterion, '{}/{}/loss{}.pth'.format(args.output_dir, args.stamp, e+1))
        
        # Step Learning Rate
        writer.add_scalar('Learning Rate', lr_scheduler.get_last_lr()[0], e)
        lr_scheduler.step()


    return history