import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv

def read_res(file_path):
    l1s, l2s, linfs = [], [], []
    with open(file_path) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            l1s.append(float(row[0]))
            l2s.append(float(row[1]))
            linfs.append(float(row[2]))

    return l1s, l2s, linfs

l1s_cat, l2s_cat, linfs_cat = read_res('res/reconstruct_cat_from_dogs.csv')
l1s_dog, l2s_dog, linfs_dog = read_res('res/reconstruct_dog_from_cats.csv')


def set_axis_style(ax, labels):
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels, rotation=45)
    ax.set_xlim(0.25, len(labels) + 0.75)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
style = {'showmedians': True}
parts1 = ax1.violinplot([l1s_cat, l1s_dog], **style)
parts2 = ax2.violinplot([l2s_cat, l2s_dog], **style)
parts3 = ax3.violinplot([linfs_cat, linfs_dog], **style)

### Coloring

for parts in [parts1, parts2, parts3]:
    for i, pc in enumerate(parts['bodies']):
        if i == 1:
            pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')

    for partname in ('cbars','cmins','cmaxes','cmedians'):
        vp = parts[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(1)

### Labels

ax1.set_ylabel(r'$\ell_1$ reconstruction error')
ax2.set_ylabel(r'$\ell_2$ reconstruction error')
ax3.set_ylabel(r'$\ell_{\infty}$ reconstruction error')

labels = [r'Dogs $\to$ cat', r'Cats $\to$ dog']
for ax in [ax1, ax2, ax3]:
    set_axis_style(ax, labels)

# matplotlib.use('pgf')
matplotlib.rcParams.update({
    'pgf.texsystem': 'pdflatex',
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
fig.tight_layout()
plt.savefig('figs/reconstruction.pgf', transparent=True, bbox_inches='tight')
