from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='norm_dist_cuda',
      version='2.0.0',
      description='A CUDA library for the calculation of general L_p-distance networks, supporting automatic differentiation.',
      url='https://github.com/zbh2047/L_inf-dist-net-v2',
      author='Bohang Zhang',
      ext_modules=[CUDAExtension('norm_dist_cuda', include_dirs=['.'],
                                 sources=['core/cuda/norm_dist.cpp',
                                          'core/cuda/norm_dist_cuda.cu',
                                          'core/cuda/inf_dist_cuda.cu',
                                          'core/cuda/bound_inf_dist_cuda.cu'])],
      cmdclass={'build_ext': BuildExtension})