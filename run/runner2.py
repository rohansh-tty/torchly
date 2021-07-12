import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import torch
from collections import defaultdict
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from .metric_monitor import MetricMonitor

tb = SummaryWriter()

def train(model, config, scheduler, epoch):
    train_metric_monitor = MetricMonitor(config.trainloader)

    model.train()
    pbar = tqdm(config.trainloader)
    correct = 0
    processed = 0
    optimizer = getattr(torch.optim, config.optimizer)(model.parameters(), **config.optimizer_params[config.optimizer])
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(config.device), target.to(config.device)

        optimizer.zero_grad()
        y_pred = model(data)

        loss = F.nll_loss(y_pred, target)
        if config.L1Lambda:
            l1 = 0
            for p in model.parameters():
                l1 += p.abs().sum()
            loss +=  1e-5 * l1

        

        loss.backward()
        optimizer.step()

        # lr changes
        scheduler.step()

        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        # update metrics
        train_metric_monitor.update('Loss', loss.item())
        train_metric_monitor.update('Accuracy', 100*correct/processed)

        lossval = train_metric_monitor.getlastval('Loss')
        accval = train_metric_monitor.getlastval('Accuracy')

        pbar.set_description(
            desc=f'TrainSet: Loss={lossval} Batch_id={batch_idx} Accuracy={accval:0.2f}')
        

    return train_metric_monitor

def test(model, config, epoch):
    test_metric_monitor = MetricMonitor(config.testloader)
    
    model.eval()
    pbar = tqdm(config.testloader)
    test_loss_value = 0
    correct = 0
    test_misc_images = []
    count = 0
    img, label = next(iter(config.testloader))
    test_input = img.to(config.device)
    
    with torch.no_grad():
        for data, target in config.testloader:
          count += 1
          data, target = data.to(config.device), target.to(config.device)
          output = model(data)
          # sum up batch loss
          loss = F.nll_loss(output, target, reduction='sum').item()
          test_loss_value += loss
        
          # get the index of the max log-probability
          pred = output.argmax(dim=1, keepdim=True)
          correct += pred.eq(target.view_as(pred)).sum().item()
          result = pred.eq(target.view_as(pred))

          accuracy = 100*correct/len(config.testloader.dataset)

          # update test metrics
          test_metric_monitor.update('Loss', loss)
          test_metric_monitor.update('Accuracy', accuracy)
          
          lossval = test_metric_monitor.getlastval('Loss')
          accval = test_metric_monitor.getlastval('Accuracy')

          if config.misclassified:
            if count > 4  and count < 15:
                for i in range(0, config.testloader.batch_size):
                    if not result[i]:
                        test_misc_images.append({'pred': list(pred)[i], 'label': list(target.view_as(pred))[i], 'image': data[i]})
            

    print('TestSet',test_metric_monitor)
    return test_metric_monitor, test_misc_images


def run(model, config):
  model_results = defaultdict()

  misclassified=None

  optimizer = getattr(torch.optim, config.optimizer)(model.parameters(), **config.optimizer_params[config.optimizer])
  scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                max_lr=0.2,
                                                steps_per_epoch=len(config.trainloader),
                                                epochs=config.EPOCHS) 

  lr_list = []

  print('='*10+'RUNNING THE MODEL'+'='*10)
  for epoch in range(config.EPOCHS):
      print('\nEPOCH {} | LR {}: '.format(epoch+1, scheduler.get_last_lr()))
      lr_list.append(scheduler.get_last_lr())
      train_metric_monitor = train(model, config, scheduler, epoch)
      test_metric_monitor, test_misc_images = test(model, config, epoch)

      lr = np.array(scheduler.get_last_lr())
      tb.add_scalar('Learning Rate', lr, epoch)
     

  torch.save(model.state_dict(), f"{config.name}.pth")
  misclassified = test_misc_images
  model_results['TrainLoss'] = train_metric_monitor['Loss']['val_list']
  model_results['TestLoss'] = test_metric_monitor['Loss']['val_list']
  model_results['TrainAcc'] = train_metric_monitor['Accuracy']['val_list']
  model_results['TestAcc'] = test_metric_monitor['Accuracy']['val_list']
  model_results['LR'] = lr_list

  return model_results, test_misc_images


def get_class_accuracy(model, config):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
      for data in config.testloader:
          images, labels = data
          labels=labels.to(config.device)
          outputs = model(images.to(config.device))
          _, predicted = torch.max(outputs, 1)
          c = (predicted == labels).squeeze()
          for i in range(4):
              label = labels[i]
              class_correct[label] += c[i].item()
              class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            config.classes[i], 100 * class_correct[i] / class_total[i]))