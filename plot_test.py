import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
test_list = [
pd.read_csv("Data/TestingData/testing1.csv", header=0, sep=';'),
pd.read_csv("Data/TestingData/testing2.csv", header=0, sep=';'),
pd.read_csv("Data/TestingData/testing3.csv", header=0, sep=';'),
pd.read_csv("Data/TestingData/testing4.csv", header=0, sep=';'),
pd.read_csv("Data/TestingData/testing5.csv", header=0, sep=';'),
pd.read_csv("Data/TestingData/testing7.csv", header=0, sep=';'),
pd.read_csv("Data/TestingData/testing6.csv", header=0, sep=';'),
pd.read_csv("Data/TestingData/testing8.csv", header=0, sep=';'),
pd.read_csv("Data/TestingData/testing9.csv", header=0, sep=';'),
pd.read_csv("Data/TestingData/testing10.csv", header=0, sep=';'),
pd.read_csv("Data/TestingData/testing11.csv", header=0, sep=';'),
pd.read_csv("Data/TestingData/testing12.csv", header=0, sep=';'),
pd.read_csv("Data/TestingData/testing13.csv", header=0, sep=';'),
pd.read_csv("Data/TestingData/testing14.csv", header=0, sep=';'),
pd.read_csv("Data/TestingData/testing15.csv", header=0, sep=';'),
pd.read_csv("Data/TestingData/testing16.csv", header=0, sep=';'),
pd.read_csv("Data/TestingData/testing17.csv", header=0, sep=';'),
pd.read_csv("Data/TestingData/testing18.csv", header=0, sep=';'),
pd.read_csv("Data/TestingData/testing19.csv", header=0, sep=';'),
pd.read_csv("Data/TestingData/testing20.csv", header=0, sep=';'),
pd.read_csv("Data/TestingData/testing21.csv", header=0, sep=';'),
pd.read_csv("Data/TestingData/testing22.csv", header=0, sep=';'),
pd.read_csv("Data/TestingData/testing23.csv", header=0, sep=';')]

fig, axes = plt.subplots(nrows=5, ncols = 5)

for i in range(len(test_list)):
    if i < 5:
        test_list[i].plot(y = ["Current"], use_index=True, ax=axes[0,i])
    if i> 4 and i < 10:
        test_list[i].plot(y = ["Current"], use_index=True, ax=axes[1,i-5])
    if i> 9 and i < 15:
        test_list[i].plot(y = ["Current"], use_index=True, ax=axes[2,i-10])
    if i> 14 and i < 20:
        test_list[i].plot(y = ["Current"], use_index=True, ax=axes[3,i-15])
    if i> 19 and i < 25:
        test_list[i].plot(y = ["Current"], use_index=True, ax=axes[4,i-20])

    

plt.show()