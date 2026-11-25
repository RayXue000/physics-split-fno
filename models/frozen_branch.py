# -*- coding: utf-8 -*-
"""旧类名兼容：线性冻结分支现由通用物理步进模块实现。"""

from models.physics_steps import LinearSpectralStep


class FrozenPhysicsBranch(LinearSpectralStep):
    """兼容 v4 构造签名 ``FrozenPhysicsBranch(symbol_fn, dt, length)``。"""

    pass
