import pandas as pd
import torch

data = torch.load("./weight/new_transE.pt")
for k,v in data.items():
    print(k,v)