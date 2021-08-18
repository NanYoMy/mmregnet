# ---- voxelmorph ----
# unsupervised learning for image registration

from . import generators
from . import cm_generators
from . import py
from .py.utils import default_unet_features


# import backend-dependent submodules
backend = py.utils.get_backend()

if backend == 'pytorch':
    # the pytorch backend can be enabled by setting the VXM_BACKEND
    # environment var to "pytorch"
    try:
        import torch
    except ImportError:
        raise ImportError('Please install pytorch to use this voxelmorph backend')

    from . import torch
    from .torch_vm import layers
    from .torch_vm import networks
    from .torch_vm import losses

else:
    # tensorflow is default backend
    try:
        import tensorflow
    except ImportError:
        raise ImportError('Please install tensorflow to use this voxelmorph backend')

    from . import tf
    from .tf_vm import layers
    from .tf_vm import networks
    from .tf_vm import losses
    from .tf_vm import utils
