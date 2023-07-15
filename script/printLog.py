import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fig, ax = plt.subplots()
ax.set_title("")
ax.set_ylim([10, 100])

log = pd.read_csv('4.csv', header=None)

# time = 60*log[0].to_numpy() + log[1].to_numpy()
time = log[2].to_numpy()

# diff = np.diff(time)
# print(diff)
# log[1].plot(ax=ax)
ax.grid()
plt.plot(time)
plt.show()
