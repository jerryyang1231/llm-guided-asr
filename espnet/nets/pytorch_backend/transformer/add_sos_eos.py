#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Unility functions for Transformer."""

import torch


def add_sos_eos(ys_pad, sos, eos, ignore_id, repeat=1, pad_input_with_eos=True):
    """Add <sos> and <eos> labels.

    :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
    :param int sos: index of <sos>
    :param int eos: index of <eos>
    :param int ignore_id: index of padding
    :return: padded tensor (B, Lmax)
    :rtype: torch.Tensor
    :return: padded tensor (B, Lmax)
    :rtype: torch.Tensor
    """
    from espnet.nets.pytorch_backend.nets_utils import pad_list

    _sos = ys_pad.new([sos]).tile(repeat)
    _eos = ys_pad.new([eos]).tile(repeat)
    ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
    ys_in = [torch.cat([_sos, y], dim=0) for y in ys]
    ys_out = [torch.cat([y, _eos], dim=0) for y in ys]

    if pad_input_with_eos:
        return pad_list(ys_in, eos), pad_list(ys_out, ignore_id)
    else:
        return pad_list(ys_in, ignore_id), pad_list(ys_out, ignore_id)
