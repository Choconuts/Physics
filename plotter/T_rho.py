import matplotlib.pyplot as plt
import numpy as np


n_sample = 500
max_val = 10.0
dx = max_val / n_sample
x = np.linspace(0, max_val, n_sample)
y = np.exp(-x)




plt.figure(figsize=[4, 2])
ax = plt.subplot(111)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))
plt.xticks([-1, 1], ['-1', '1'])
plt.yticks([0, 1], ['0', '1'])
plt.plot(x, y, color='blue', label='T')
plt.legend()
plt.show()



if __name__ == '__main__':
    pass
