#AppliedAIcourse.com
import numpy as np
import matplotlib.pyplot as plt

def plot_distribution(labels, t_cv_te='train'):
    '''
    This function prints the distribution of output variable in a given dataframe and also prints the stats
    '''
    print('-'*80)
    class_distribution = labels.value_counts().sort_index()
    class_distribution.plot(kind='bar')
    plt.xlabel('Class')
    plt.ylabel('Data points per Class')
    plt.title('Distribution of yi in ' + t_cv_te + ' data')
    plt.grid()
    plt.show()

    # ref: argsort https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
    # -(train_class_distribution.values): the minus sign will give us in decreasing order
    sorted_yi = np.argsort(-class_distribution.values)
    for i in sorted_yi:
        print('Number of data points in class', i+1, ':',\
              class_distribution.values[i], '(', np.round((class_distribution.values[i]/labels.shape[0]*100), 3), '%)')
