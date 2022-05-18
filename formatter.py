import matplotlib.pyplot as plt

SIZE = 14

plt.rc('font', size=SIZE)
plt.rc('axes', titlesize=SIZE)
plt.rc('axes', labelsize=SIZE)
plt.rc('xtick', labelsize=SIZE)
plt.rc('ytick', labelsize=SIZE)
plt.rc('legend', fontsize=SIZE)
plt.rc('figure', titlesize=SIZE)


def export(filename):
    plt.savefig(f'images/{filename}.png', bbox_inches='tight', pad_inches=0, dpi=600)
