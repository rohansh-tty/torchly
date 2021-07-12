from collections import defaultdict

class MetricMonitor:
    def __init__(self, dataloader, float_precision=4):
        self.float_precision = float_precision
        self.set()
        self.dataloader = dataloader
        
    def set(self):
        self.metrics = defaultdict(lambda: {'val':0, 'count': 0, 'avg': 0, 'val_list': []})
        
    def update(self, metric, value):
        metric = self.metrics[metric]
        
        metric['val'] += value
        metric['count'] += 1
        metric['val_list'].append(value)
        
        
    def __str__(self):
        string1 = ' || '.join([
            f"{metric_name}: {round(self.getaverage(metric_name), self.float_precision)}"
            for metric_name, metric_prop in self.metrics.items()
        ])
        
        return  string1
        
    def __getitem__(self, metric):
        return self.metrics[metric]

    
    
    def getlastval(self, metric):
        return self.metrics[metric]['val_list'][-1]

    def getaverage(self, metric):
        val_list = self.metrics[metric]['val_list']
        multiplier = 100 if metric=='Accuracy' else 1
        return multiplier*sum(val_list)/len(self.dataloader.dataset)
    
        