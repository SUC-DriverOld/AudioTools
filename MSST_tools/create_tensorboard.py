import numpy as np
from torch.utils.tensorboard import SummaryWriter

epochs = []
training_loss = []
instr_sdr_aspiration = []
instr_sdr_other = []
sdr_avg = []
learning_rate = []

patterns = {
    "Training loss:": training_loss,
    "Instr SDR aspiration:": instr_sdr_aspiration,
    "Instr SDR other:": instr_sdr_other,
    "SDR Avg:": sdr_avg
}

with open('log.txt', 'r') as file:
    for line in file:
        if "Train epoch:" in line and "Learning rate:" in line:
            epoch_part, lr_part = line.split("Learning rate:")
            epochs.append(int(epoch_part.split("Train epoch:")[-1].strip()))
            learning_rate.append(float(lr_part.strip()))
        else:
            for key, value_list in patterns.items():
                if key in line:
                    value = float(line.split(key)[-1].strip())
                    value_list.append(value)

writer = SummaryWriter(log_dir='runs/training_logs')

for i, epoch in enumerate(epochs):
    if i < len(training_loss):
        writer.add_scalar('Metrics/Training Loss', training_loss[i], epoch)
    if i < len(instr_sdr_aspiration):
        writer.add_scalar('Metrics/Instr SDR Aspiration', instr_sdr_aspiration[i], epoch)
    if i < len(instr_sdr_other):
        writer.add_scalar('Metrics/Instr SDR Other', instr_sdr_other[i], epoch)
    if i < len(sdr_avg):
        writer.add_scalar('Metrics/SDR Avg', sdr_avg[i], epoch)
    if i < len(learning_rate):
        writer.add_scalar('Metrics/Learning Rate', learning_rate[i], epoch)

writer.close()
print("TensorBoard logs have been written to the 'runs/training_logs' directory.")
