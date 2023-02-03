python fancy_plot.py --data=mnist_38 --labels=standard_sdr_max_1 --figsize=small --clear_figs --no_x_label
python fancy_plot.py --data=malimg --labels=standard_sdr_large_4 --figsize=small --clear_figs --no_x_label --no_y_label --x_log
python fancy_plot.py --data=fashion_mnist_shirts --labels=standard_1 --figsize=small --clear_figs
python fancy_plot.py --data=cifar10_catsdogs --labels=standard_2 --figsize=small --no_y_label --clear_figs 

cp -R figs/* ~/ResearchRepos/crpaper/figs_gen/

