import matplotlib.pyplot as plt
import csv


figure, axis = plt.subplots(1, 2)

files = ['/home/cnhzcy14/work/project/script/0.csv',
         '/home/cnhzcy14/work/project/script/1.csv',
         '/home/cnhzcy14/work/project/script/2.csv']


ms = []
fps = []

for i in range(len(files)):
    ms.append([])
    fps.append([])
    with open(files[i], newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            ms[i].append(float(row[0]))
            fps[i].append(float(row[1]))
    frame = list(range(len(ms[i])))
    axis[0].plot(frame, ms[i], alpha=0.5)
    axis[1].plot(frame, fps[i], alpha=0.5)

axis[0].set_title("ms")
axis[0].set_ylim([0, 100])
axis[1].set_title("fps")
axis[1].set_ylim([0, 100])

plt.show()
