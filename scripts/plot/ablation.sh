python fancy_plot.py --data=cifar10_catsdogs --experiment=ablation --labels=ablation_feature_map --figsize=small --title="Cats-dogs (certifying cats)" --clear_figs
python fancy_plot.py --data=cifar10_dogscats --experiment=ablation --labels=ablation_feature_map --figsize=small --title="Dogs-cats (certifying dogs)" --no_y_label --clear_figs

python fancy_plot.py --data=cifar10_catsdogs --experiment=ablation --labels=ablation_reg --figsize=large

cp -R figs/* ~/ResearchRepos/crpaper/figs_gen/

