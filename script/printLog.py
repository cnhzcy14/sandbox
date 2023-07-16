import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fig, ax = plt.subplots()
ax.set_title("")
ax.set_ylim([0, 300])

csv_file = input('Input a csv file:\n')


log = pd.read_csv(csv_file, header=None)

host_time = (60*log[0].to_numpy() + log[1].to_numpy()) * 1000

device_time_0 = log[4].to_numpy()
device_time_1 = log[5].to_numpy()

# time = log[2].to_numpy()

# diff = np.diff(time)
# print(diff)
# log[1].plot(ax=ax)
ax.grid()
plt.plot(np.diff(host_time, n=1, axis=-1))
plt.show()
