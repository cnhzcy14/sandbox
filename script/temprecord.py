import numpy as np
import matplotlib.pyplot as plt

cpu = []
gpu = []
with open('cpu_therm') as f:
	for line in f:
		data = line
		cpu.append(float(data))

with open('gpu_therm') as f:
	for line in f:
		data = line
		gpu.append(float(data))

line_cpu, = plt.plot(cpu, label='CPU')
line_gpu, = plt.plot(gpu, label='GPU')
plt.ylabel('temperature')
plt.legend(handles=[line_cpu, line_gpu])

plt.show()