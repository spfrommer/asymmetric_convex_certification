## Install
This code was developed for Python 3.7.5 and is easiest to reproduce using [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/). After setting up the virtual environment, simply run `bash setup.sh`.

For the linfinity distance nets:
- python setup.py install --user
- pip install -e .
- make sure CUDA_HOME is set to your cuda install

Reproducing the convex combination experiments in Appendix C requires a working MOSEK install and license as documented [here](https://docs.mosek.com/latest/install/installation.html) -- this license is freely available for academic researchers.

## Key implementations
The base class for all convexly certified methods lies in `convexrobust/model/convex_certifiable.py`, with specific subclass instantiations lying in `convexrobust/model/insts/convex.py`. The file `convexrobust/model/modules.py` contains implementations of ICNN MLPs and convnets.

## Data
All datasets are downloaded automatically upon running.

## Execution
To reproduce the main experiments of the paper, change to the `convexrobust/main` directory and run `python main.py --data=mnist_38 --train`, where the `data` option is one of `mnist_38`, `fashion_mnist_shirts`, `malimg`, or `cifar10_catsdogs`. The figure papers can then be produced with `python plot.py --data=mnist_38 --labels=mnist_38_paper` (change accordingly) and should lie in `convexrobust/main/figs/`.

To reproduce the convex combination experiments, enter the `scripts/misc` directory and execute `reconstruction_main.py` and `reconstruction_plot.py`. The same directory also contains the malimg multiclass experiments and the mnist sweep experiments from the Appendix.

For convenience, `scripts/misc/simple_example.py` contains a minimal script to train and certify a convex network.

### Running abCROWN
After running the above, we can execute the abCROWN verifier for the abCROWN baseline, which runs separately. First, install as follows:

```
cd lib/alpha-beta-CROWN
conda env create -f complete_verifier/environment.yml --name alpha-beta-crown2
conda activate alpha-beta-crown2
conda develop .
cd ../..
bash setup_abcrown.sh
```
Note that we need to run a different install script to be compatible with the alpha-beta-CROWN package versions.

conda deactivate
conda remove -n alpha-beta-crown2 --all

After the abCROWN install has finished, simply run `bash scripts/abcrown/xxx.sh` from the root directory. The certified will be populated automatically into the apropriate results file. Replotting should then show the abCROWN certified accuracy curve.
