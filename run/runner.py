from tqdm import tqdm_notebook
from tqdm import tqdm
from collections import defaultdict 

class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.metrics = defaultdict()
        self.reset()
 
    def __getitem__(self, metric_name):
      return self.metrics[metric_name]
 
 
    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0, "list":[]})
 
    def update(self, metric_name, val):
        metric = self.metrics[metric_name]
 
        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]
        metric["list"].append(val)
 
    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )
 
      
      
 
def train(train_loader, model, modelconfig, criterion, optimizer, scheduler, epoch):
    train_metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)
    correct, processed, eqsum_list, total_loss = 0, 0, [], 0
    for i, (images, target) in enumerate(stream, start=1):
        
        image = images['image'].to(modelconfig.device, non_blocking=True)
        target = target.to(modelconfig.device)
        output = model(image)
        loss = criterion(output, target)
        pred = output.argmax(dim=1, keepdim=False)
       
        eqsum = pred.eq(target.view_as(pred)).sum().item()
        correct += eqsum
        processed += len(images)
        eqsum_list.append(eqsum)
        accuracy = 100*correct/(128*processed)
        total_loss += loss
 
        train_metric_monitor.update("Loss", loss.item())
        train_metric_monitor.update("Accuracy", accuracy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        stream.set_description(
            "   Train.      {metric_monitor}".format(epoch=epoch, metric_monitor=train_metric_monitor)
        )
    return train_metric_monitor
 
 
def validate(val_loader, model, modelconfig, criterion,epoch):
    val_metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    correct, processed = 0, 0 
    test_misc_images = []
    count = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(stream, start=1):
            count += 1
            images = images['image'].to(modelconfig.device, non_blocking=True)
            target = target.to(modelconfig.device)
            output = model(images)
            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=False).squeeze(dim=0)
            correct += pred.eq(target.view_as(pred)).sum().item()
            result = pred.eq(target.view_as(pred))
            
            processed += (len(images))
            accuracy = 100*correct/(processed)
 
            if count > 4  and count < 15 and images.shape[0] == modelconfig.testloader.batch_size:
              for i in range(0, modelconfig.testloader.batch_size):
                  if not result[i]:
                    
                    test_misc_images.append({'pred': list(pred)[i], 'label': list(target.view_as(pred))[i], 'image': images[i]})
 
            val_metric_monitor.update("Loss", loss.item())
            val_metric_monitor.update("Accuracy", accuracy)
            stream.set_description(
                "   Validation. {metric_monitor}".format(epoch=epoch, metric_monitor=val_metric_monitor)
            )
    return val_metric_monitor, test_misc_images
 
