import matplotlib.pyplot as plt
import os

data_pth = "/home/liuk/data/kesci/train_data/train_list.txt"

X_data = []
X_data_sum = {}
Y_data = []
with open(data_pth,'r') as f:
    file = f.readlines()
    for line in file:
        id = int(line.split()[-1])
        if id not in X_data:
            X_data.append(int(id))
            X_data_sum[id] = 1
        else:
            X_data_sum[id] +=1
#print(len(X_data))
for i in range(len(X_data)):
    number = X_data_sum[i]
    print(i,number)
    # if i in [1055,1477]:
    #     print(i,X_data_sum[i],"数据过大")
    #     X_data_sum[i] = 6
    Y_data.append(X_data_sum[i])
#print(X_data_sum[1477])
plt.figure()
plt.bar(X_data,Y_data)
plt.title("data distribution")
plt.savefig("data.png")
