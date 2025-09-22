#%% import libraries
import matplotlib.pyplot as plt
import numpy as np
import bemcs
import pandas as pd

#%% Read source file and mesh of the domain
fileinput = "HeterogeneousDomainMesh.csv"

datain = pd.read_csv(fileinput)

x1 = datain["x1"].values
x2 = datain["x2"].values
y1 = datain["z1"].values
y2 = datain["z2"].values
BCtype = datain["BC_type"].values
BCval = datain["value"].values
# %%
