import matplotlib as mpl
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt

data=pd.read_csv('loss.csv')
x=data['step']
y1=data['s=2']
y2=data['s=1.05']
y3=data['s=0.8']
y4=data['s=0.9']
y5=data['s=1.2']
y6=data['s=1.1']
y7=data['s=1.11']
y8=data['p']
plt.figure()
plt.title('Result Analysis')
plt.subplot(8,1,1)
plt.plot(x, y8, color='yellowgreen', label='paper')
plt.legend()
plt.xlabel('step')
plt.ylabel('loss')
plt.subplot(8,1,2)
plt.plot(x, y1, color='green', label='s=2')
plt.legend()
plt.subplot(8,1,3)
plt.plot(x, y2, color='red', label='s=1.05')
plt.legend()
plt.subplot(8,1,4)
plt.plot(x, y3, color='skyblue', label='s=0.8')
plt.legend()
plt.subplot(8,1,5)
plt.plot(x, y4, color='blue', label='s=0.9')
plt.legend()
plt.subplot(8,1,6)
plt.plot(x, y5, color='yellow', label='s=1.2')
plt.legend()
plt.subplot(8,1,7)
plt.plot(x, y6, color='pink', label='s=1.1')
plt.legend()  # 显示图例
plt.subplot(8,1,8)
plt.plot(x, y7, color='black', label='s=1.1')
plt.legend()  # 显示图例
plt.xlabel('step')
plt.ylabel('loss')
plt.show()