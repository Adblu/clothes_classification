import matplotlib.pyplot as plt

from src.utils.helpers import bring_stats

if __name__ == "__main__":
    file = '/home/n/Documents/STER/src/models/clothes_experiment/20210715_130634.log.json'
    file2 = '/home/n/Documents/STER/src/models/clothes_experiment_tf/20210716_063252.log.json'

    accuracy = bring_stats(file)
    accuracy2 = bring_stats(file2)

    ax0 = plt.subplot(111)

    ax0.plot(range(len(accuracy)), accuracy, 'r*-')
    ax0.plot(range(len(accuracy2)), accuracy2, 'g*-')

    ax0.legend(['baseline', 'tf'])
    ax0.set_title('Bbox')
    ax0.set_ylabel('AP')
    ax0.set_xlabel('Epoch')

    ax0.grid()
    plt.savefig('../figures/experiment_summary.png')
