source ~/.bashrc  # Source the add2virtualenv command

pip install -r requirements_abcrown.txt
python setup_abcrown.py develop
pip install git+https://github.com/facebookresearch/jacobian_regularizer
conda develop lib/smoothingSplittingNoise/
