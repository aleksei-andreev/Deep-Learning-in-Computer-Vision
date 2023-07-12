#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import sys
from yolox.exp import Exp as MyExp

MAIN_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(MAIN_PATH))

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.0
        self.width = 1.0
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
