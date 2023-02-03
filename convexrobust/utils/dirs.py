import os.path as op


def path(*path: str) -> str:
    return op.join(*path)


ROOT_DIR = op.dirname(op.dirname(op.dirname(op.abspath(__file__))))
def root_path(*path: str) -> str:
    return op.join(ROOT_DIR, *path)


OUT_DIR = root_path('out')
def out_path(*path: str) -> str:
    return op.join(OUT_DIR, *path)


DATA_DIR = root_path('data')
def data_path(*path: str) -> str:
    return op.join(DATA_DIR, *path)


PRETRAIN_DIR = root_path('pretrain')
def pretrain_path(*path: str) -> str:
    return op.join(PRETRAIN_DIR, *path)


LIB_DIR = root_path('lib')
def lib_path(*path: str) -> str:
    return op.join(LIB_DIR, *path)
