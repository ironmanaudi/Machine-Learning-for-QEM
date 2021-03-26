import torch 
import numpy as np
import matplotlib.pyplot as plt

mitigated = [0.003890975824826187,0.0011680859979919334,0.0003599496867570662,0.006865552872544675,0.002667532265400907]
noisy = [0.02792040305554322,0.011623957833124445,0.008363536876754233,0.024753818252132877,0.020315605618278394]

d = [4,5,6,7,8]

plt.title('different qubits')
plt.plot(d, mitigated, color='green', label='mitigated')
plt.plot(d, noisy, color='red', label='noisy')

plt.legend() # 显示图例

plt.xlabel('number of qubits')
plt.ylabel('error rates')
#plt.show()
plt.savefig('./diff_num.png')




