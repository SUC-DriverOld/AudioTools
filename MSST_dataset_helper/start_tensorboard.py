import re
import os
import numpy as np
from tensorboardX import SummaryWriter

writer = SummaryWriter('runs/metrics_visualization')

epoch_pattern = re.compile(r'Train epoch: (\d+) Learning rate: ([\d.eE+-]+)')
training_loss_pattern = re.compile(r'Training loss: ([\d.]+)')
metric_pattern = re.compile(r'(\w+ \w+ \w+): ([\d.]+) \(Std: ([\d.]+)\)')
avg_metric_pattern = re.compile(r'Metric avg (\w+)\s+: ([\d.]+)')

data = {
    'common': {
        'learning_rate': [],
        'training_loss': []
    },
    'dry': {
        'Instr dry sdr': [],
        'Instr dry l1_freq': [],
        'Instr dry si_sdr': []
    },
    'other': {
        'Instr other sdr': [],
        'Instr other l1_freq': [],
        'Instr other si_sdr': []
    },
    'avg': {
        'Metric avg sdr': [],
        'Metric avg l1_freq': [],
        'Metric avg si_sdr': []
    }
}

std_data = {
    'dry': {key: [] for key in data['dry'].keys()},
    'other': {key: [] for key in data['other'].keys()}
}

# 读取数据文件
with open(r'E:\AI\datasets\msst\train.log', 'r') as f:
    epoch = -1
    for line in f:
        epoch_match = epoch_pattern.match(line)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            learning_rate = float(epoch_match.group(2))
            data['common']['learning_rate'].append((epoch, learning_rate))
            continue
        
        training_loss_match = training_loss_pattern.match(line)
        if training_loss_match:
            training_loss = float(training_loss_match.group(1))
            data['common']['training_loss'].append((epoch, training_loss))
            continue
        
        metric_match = metric_pattern.match(line)
        if metric_match:
            metric_name = metric_match.group(1)
            metric_value = float(metric_match.group(2))
            std_value = float(metric_match.group(3))
            
            # 保存值和标准差到正确的类别
            if metric_name in data['dry']:
                data['dry'][metric_name].append((epoch, metric_value))
                std_data['dry'][metric_name].append((epoch, std_value))
            elif metric_name in data['other']:
                data['other'][metric_name].append((epoch, metric_value))
                std_data['other'][metric_name].append((epoch, std_value))
            continue
        
        avg_metric_match = avg_metric_pattern.match(line)
        if avg_metric_match:
            avg_metric_name = f'Metric avg {avg_metric_match.group(1)}'
            avg_metric_value = float(avg_metric_match.group(2))
            data['avg'][avg_metric_name].append((epoch, avg_metric_value))

for category, metrics in data.items():
    for key, values in metrics.items():
        category_path = f'{category}/{key.replace(" ", "_").lower()}'
        for epoch, value in values:
            writer.add_scalar(f'{category_path}', value, epoch)

for category, metrics in std_data.items():
    for key, values in metrics.items():
        category_path = f'{category}/{key.replace(" ", "_").lower()}_std'
        for epoch, std in values:
            writer.add_scalar(f'{category_path}', std, epoch)

writer.close()
os.system('tensorboard --logdir=runs')
