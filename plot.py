import matplotlib.pyplot as plt
import numpy as np

# r1 = np.loadtxt("rewards_sarsa.txt")
# r2 = np.loadtxt("rewards_qlear.txt")

r1 = np.loadtxt("up.txt")
r2 = np.loadtxt("down.txt")
r3 = np.loadtxt("right.txt")
r4 = np.loadtxt("left.txt")

# plt.plot(r1, 'r-', label='lin-sto-sarsa')
# plt.plot(r2, 'b-', label='lin-sto-q')

plt.plot(r1, 'r-', label='up')
plt.plot(r2, 'b-', label='down')
plt.plot(r3, 'g-', label='right')
plt.plot(r4, 'c-', label='left')

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
plt.show()
