import torch 
import numpy as np
import matplotlib.pyplot as plt

mitigated = [0.003890975824826187,0.0011680859979919334,0.0003599496867570662,9.176781774058537e-05,0.002667532265400907]
noisy = [0.02792040305554322,0.011623957833124445,0.008363536876754233,0.0067634257363564584,0.020315605618278394]
ratio = []

for a,b in zip(mitigated, noisy):
    ratio.append(a/b)


d = [4,5,6,7,8]

#plt.title('RNN Error Ratio')
plt.bar(range(len(d)), ratio,color='skyblue',tick_label=d)


plt.xlabel('Number of Qubits')
plt.ylabel('Error Mitigated / Error Unmitigated')
#plt.show()
plt.savefig('./diff_num_rate.png')




