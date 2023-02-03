source ~/.bashrc  # Source the add2virtualenv command

pip install -e .
pip install git+https://github.com/facebookresearch/jacobian_regularizer
add2virtualenv lib/smoothingSplittingNoise/
