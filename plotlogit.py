import matplotlib.pyplot as plt
import torch 
import numpy
import os


Folder = 'ABMILattention_logits/TCGA-L9-A8F4-01Z-00-DX1.E2BBB8DE-94E2-4781-9B55-8A4CFBF8A69D/'
File = os.path.join(Folder, '49.pt')
logit = torch.load(File).cpu()
logit = logit - logit.mean()
logit = logit[0,0]
sig_logit = torch.sigmoid(logit)
exp_logist = torch.exp(logit)

logit_max = logit.max()
logit_min = logit.min()

Xaxis = torch.tensor([i/1000 for i in range(1000)])*(logit_max-logit_min)+logit_min

fig, ax1 = plt.subplots()
ax1.hist(logit, 500)
ax1.set_ylabel('Number per bin')
ax1.set_xlabel('attention logit')
ax2 = ax1.twinx()



ax2.plot(Xaxis, torch.sigmoid(Xaxis), c='b', label = 'Sigmoid')
ax2.plot(Xaxis, torch.exp(Xaxis), c='r', label = 'Exponential')
ax2.legend()
ax2.set_xlabel('attention logit')
ax2.set_ylabel('function value before normalization')
ax2.set_ylim(0, 2)



plt.show()