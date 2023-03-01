import matplotlib.pyplot as plt
import csv

figure, axis = plt.subplots(3, 4)
files = ['/home/cnhzcy14/work/project/script/0.csv',
         '/home/cnhzcy14/work/project/script/1.csv']

colNum = 12
data = []
for i in range(colNum):
    data.append([])

for i in range(len(files)):
    for col in range(colNum):
        data[col].append([])

    with open(files[i], newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            for col in range(colNum):
                data[col][i].append(float(row[col]))
    x = list(range(len(data[0][i])))
    for col in range(colNum):
        axis[col//4, col%4].plot(x, data[col][i], alpha=0.5, label=str(i))

axis[0, 0].set_title("")
axis[0, 0].set_ylim([0, 300])
axis[0, 0].grid()
axis[0, 0].legend()

axis[0, 1].set_title("")
axis[0, 1].set_ylim([0, 30])
axis[0, 1].grid()

axis[0, 2].set_title("")
axis[0, 2].set_ylim([0, 30])
axis[0, 2].grid()

axis[0, 3].set_title("")
axis[0, 3].set_ylim([0, 30])
axis[0, 3].grid()

axis[1, 0].set_title("")
axis[1, 0].set_ylim([0, 30])
axis[1, 0].grid()

axis[1, 1].set_title("")
axis[1, 1].set_ylim([0, 30])
axis[1, 1].grid()

axis[1, 2].set_title("")
axis[1, 2].set_ylim([0, 30])
axis[1, 2].grid()

axis[1, 3].set_title("")
axis[1, 3].set_ylim([0, 30])
axis[1, 3].grid()

axis[2, 0].set_title("")
axis[2, 0].set_ylim([0, 30])
axis[2, 0].grid()

axis[2, 1].set_title("")
axis[2, 1].set_ylim([0, 30])
axis[2, 1].grid()

axis[2, 2].set_title("")
axis[2, 2].set_ylim([0, 30])
axis[2, 2].grid()

axis[2, 3].set_title("")
axis[2, 3].set_ylim([0, 100])
axis[2, 3].grid()

# figure.suptitle("Main Title")
plt.show()