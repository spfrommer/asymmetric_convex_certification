python fancy_plot.py --data=mnist_38 --figsize=small --no_x_label --randsmooth_sweep
python fancy_plot.py --data=malimg --figsize=small --no_x_label --no_y_label --randsmooth_sweep 
python fancy_plot.py --data=fashion_mnist_shirts --figsize=small --randsmooth_sweep 
python fancy_plot.py --data=cifar10_catsdogs --figsize=small --no_y_label --randsmooth_sweep  

cp -R figs/* ~/ResearchRepos/crpaper/figs_gen/

