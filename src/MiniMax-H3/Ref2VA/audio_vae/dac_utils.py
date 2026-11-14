# SPDX-License-Identifier: MIT
# Adapted from https://github.com/jik876/hifi-gan under the MIT license.


def init_weights(m, mean=0.0, std=0.01):
    """Initialize ``Conv*`` layers, including those wrapped with ``weight_norm``.

    BigVGAN (``dac_bigvgan.py``) wraps every conv in
    ``torch.nn.utils.parametrizations.weight_norm`` and then calls
    ``self.convs1.apply(init_weights)`` / ``self.convs2.apply(init_weights)``
    / ``self.conv_post.apply(init_weights)``. Under the parametrization
    API, ``m.weight`` is computed on access from ``m.weight_g`` and
    ``m.weight_v``, so the previous ``m.weight.data.normal_(...)`` was
    a silent no-op for the stored parameters. Branch on the
    parametrized attributes to actually update the underlying
    parameters, matching the legacy ``torch.nn.utils.weight_norm``
    behaviour that this file historically targeted.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        if hasattr(m, "weight_v") and hasattr(m, "weight_g"):
            m.weight_v.data.normal_(mean, std)
            m.weight_g.data.fill_(1.0)
        else:
            m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)
