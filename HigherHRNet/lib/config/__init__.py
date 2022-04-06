# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------
'''
__init__.py该文件的作用就是相当于把自身整个文件夹当作一个包来管理，每当有外部import的时候，就会自动执行里面的函数。

'''

from .default import _C as cfg
from .default import update_config
from .default import check_config
from .models import MODEL_EXTRAS