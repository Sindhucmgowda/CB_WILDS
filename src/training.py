import torch
import logging
import os
import torch.nn.functional as F

logger = logging.getLogger("causal_bootstrap")

def cost_fn(out, tar, weighting):
    return F.cross_entropy(out, tar, weight=weighting)

def train(args, model, writer, device, train_loader, optimizer, epoch, batch_size, n_steps = None):

    model.train() # training model 

    epoch_loss = 0
    epoch_acc = 0

    offset = len(train_loader) * (epoch-1)
    
    if n_steps is None:
        n_steps = len(train_loader)

    for batch_idx, [data, target, _ ] in enumerate(train_loader):
        
        # import pdb; pdb.set_trace()

        data = data.float().to(device) # add channel dimension
        target = target.long().to(device)

        target = target.view((-1))
        
        if args.data == "MNIST_deep":
            data = data.reshape(data.shape[0],-1)

        output = model(data) ## no softmax applied 

        if args.data == "NIH" or args.data == "MIMIC": 
            weights = torch.tensor([0.1,0.9], dtype = torch.float).to(device) 
        if args.data == "CelebA_attr": 
            weights = torch.tensor([0.2,0.8], dtype = torch.float).to(device)
        if args.data == "NIH_multi":
            weights = torch.tensor([0.1,0.8,0.1], dtype = torch.float).to(device)
        else:
            weights = None
        
        loss = cost_fn(output, target, weighting=weights)
        
        optimizer.zero_grad()
        loss.backward(retain_graph=False)
        optimizer.step()

        pred = F.softmax(output, dim=1).max(1, keepdim=True)[1] # get the index of the max log-probability
        
        acc = 1.*pred.eq(target.view_as(pred)).sum().item()/len(data)

        epoch_loss += loss.item()
        epoch_acc += acc
        
        if batch_idx % args.log_interval == 0:
            logger.info('Train Chunk: {} [{}/{} ({:.0f}%)]\tAccuracy: {:.4f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), n_steps*data.shape[0],
                100. * batch_idx / n_steps, acc, loss.item()))
        
        # index.append((batch_idx+offset))
        writer.add_scalar('Loss/train/batch', loss.item(), (batch_idx+offset))
        writer.add_scalar('Accuracy/train/batch', acc, (batch_idx+offset))
        
        if batch_idx + 1 == n_steps:
            break

    return (epoch_loss/len(train_loader)), (epoch_acc/len(train_loader))

def train_helper(args, model, writer, device, train_loader, optimizer, epoch, batch_size):

    model.train() # training model 

    epoch_loss = 0
    epoch_acc = 0

    offset = len(train_loader) * (epoch-1)

    for batch_idx, [data, target, helper] in enumerate(train_loader):
        
        data = data.float().to(device) # add channel dimension
        target = target.long().to(device)
        
        if args.type == 'back' or args.type == 'front':                      
            helper = helper.float().to(device)
            output = model(data, helper) ## no softmax applied 

        elif args.type == 'back_front' or args.type == 'par_back_front':
            conf = helper[0].float().to(device)
            med = helper[1].float().to(device)
            output = model(data, conf, med) ## no softmax applied 

        target = target.view((-1))
        
        # import pdb; pdb.set_trace()
        if args.data == "NIH" or args.data == "MIMIC": 
            weights = torch.tensor([0.1,0.9], dtype = torch.float).to(device) 
        if args.data == "CelebA_attr": 
            weights = torch.tensor([0.2,0.8], dtype = torch.float).to(device)
        else:
            weights = None
        
        loss = cost_fn(output, target, weighting=weights)
        
        optimizer.zero_grad()
        loss.backward(retain_graph=False)
        optimizer.step()

        pred = F.softmax(output, dim=1).max(1, keepdim=True)[1] # get the index of the max log-probability
        
        acc = 1.*pred.eq(target.view_as(pred)).sum().item()/len(data)

        epoch_loss += loss.item()
        epoch_acc += acc
        
        if batch_idx % args.log_interval == 0:

            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tAccuracy: {:.4f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), acc, loss.item()))
        
        # index.append((batch_idx+offset))
        writer.add_scalar('Loss/train/batch', loss.item(), (batch_idx+offset))
        writer.add_scalar('Accuracy/train/batch', acc, (batch_idx+offset))

    # pdb.set_trace()
    return (epoch_loss/len(train_loader)), (epoch_acc/len(train_loader))

def snapshot(dir_path, res_pth, run_name, state):
    snapshot_file = os.path.join(dir_path,
                    run_name + '-model_best.pth')
    
    torch.save(state, snapshot_file)
    logger.info("Snapshot saved to {}\n".format(snapshot_file))

    with open(os.path.join(res_pth, "best_model_pth.txt"), 'w') as res:
        res.write(f"Best epoch: {state['epoch']}, Best val acc: {state['validation_acc']}, Best val loss: {state['validation_loss']}, Model path: {snapshot_file}")
