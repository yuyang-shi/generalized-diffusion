"""Modified code from https://github.com/yang-song/score_sde"""
# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Common layers for defining score networks.
"""
import functools
import math
import string
from typing import Any, Sequence, Optional
from dataclasses import dataclass

import jax
import haiku as hk
import jax.nn as jnn
import jax.numpy as jnp

from score_sde.utils import register_category

get_activation, register_activation = register_category("activation")

register_activation(jnn.elu, name="elu")
register_activation(jnn.relu, name="relu")
register_activation(functools.partial(jnn.leaky_relu, negative_slope=0.01), name="lrelu")
register_activation(jnn.swish, name="swish")
register_activation(jnn.sigmoid, name="sigmoid")
register_activation(jnn.soft_sign, name="softsign")
register_activation(jnn.hard_sigmoid, name="hardsigmoid")
register_activation(jnn.hard_tanh, name="hardtanh")


register_activation(jnp.sin, name='sin')
register_activation(jnp.tanh, name='tanh')

