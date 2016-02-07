import matplotlib.pyplot as plt
import pandas
import numpy as np

class data(object):
    def __init__(self, file_path):
        self.x = []
        self.y = []
        self.label = []
        with open(file_path, 'r') as f:
            for line in f:
                self.x.append(float(line.split(',')[0]))
                self.y.append(float(line.split(',')[1]))
                self.label.append(int(line.split(',')[2]))

def main():
    file_path = "../machine-learning-ex2/ex2/ex2data1.txt"
    data1 = data(file_path)
    np_label = np.asarray(data1.label)
    np_x = np.asarray(data1.x)
    np_y = np.asarray(data1.y)
    x_neg = np_x[np_label == 0]
    y_neg = np_y[np_label == 0]
    x_pos = np_x[np_label == 1]
    y_pos = np_y[np_label == 1]
    plt.plot(x_neg, y_neg, 'bo', label='Not Admitted')
    plt.plot(x_pos, y_pos, 'ro', label='Admitted')
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.show()

if __name__ == "__main__":
    main()
