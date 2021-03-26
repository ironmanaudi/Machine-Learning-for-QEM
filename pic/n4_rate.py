import torch 
import numpy as np
import matplotlib.pyplot as plt

l4i = [0.004000701174184453,0.003890975824826187,0.0009185621961657967,0.00875265855393749]
l4n = [0.030874872029172912,0.02792040305554322,0.00260638515593162,0.056919292394370755]
l5i = [0.0030367392190860736,0.0015127863478184421,0.0006444835311717127,0.004072257551872035]
l5n = [0.01417301026638706,0.011623957833124445,0.00321365480036956,0.01499014994743188]
l6i = [0.0029209759405482627,0.00027961509891232564,0.0035096781638215752,0.0014028898578504203]
l6n = [0.03317407373784561,0.008363536876754233,0.011062528789465809,0.01434777559317088]
l7i = [0.009802726464263004,0.0031446358977457083,0.004180169848072857,0.008175268401808317]
l7n = [0.04039102285145361,0.011448843333444128,0.024753818252132877,0.024289824421465926]
l8i = [0.004117132089600481,0.004091403247225774,0.0037108080509785507,0.0007992235390034234]
l8n = [0.04019264403940389,0.020315605618278394,0.021597263695254,0.020322989832199007]
ratio8 = []
ratio10 = []
ratio12 = []
ratio14 = []

for i in range(4):
    if i==0:
        ratio8.append(l4i[i]/l4n[i])
        ratio8.append(l5i[i]/l5n[i])
        ratio8.append(l6i[i]/l6n[i])
        ratio8.append(l7i[i]/l7n[i])
        ratio8.append(l8i[i]/l8n[i])
    if i==1:
        ratio10.append(l4i[i]/l4n[i])
        ratio10.append(l5i[i]/l5n[i])
        ratio10.append(l6i[i]/l6n[i])
        ratio10.append(l7i[i]/l7n[i])
        ratio10.append(l8i[i]/l8n[i])
    if i==2:
        ratio12.append(l4i[i]/l4n[i])
        ratio12.append(l5i[i]/l5n[i])
        ratio12.append(l6i[i]/l6n[i])
        ratio12.append(l7i[i]/l7n[i])
        ratio12.append(l8i[i]/l8n[i])
    if i==3:
        ratio14.append(l4i[i]/l4n[i])
        ratio14.append(l5i[i]/l5n[i])
        ratio14.append(l6i[i]/l6n[i])
        ratio14.append(l7i[i]/l7n[i])
        ratio14.append(l8i[i]/l8n[i])


size = 5
x = np.arange(4,9)

total_width, n = 0.8, 4
width = total_width / 4
x = x - (total_width - width) / 2

plt.bar(x, ratio8,  width=width, label='Depth=8')
plt.bar(x + width, ratio10, width=width, label='Depth=10')
plt.bar(x + 2 * width, ratio12, width=width, label='Depth=12')
plt.bar(x + 3 * width, ratio14, width=width, label='Depth=14')

plt.legend()


plt.xlabel('Number of Qubits')
plt.ylabel('Error Mitigated / Error Unmitigated')
#plt.show()
plt.savefig('./n=4_rate.png')
