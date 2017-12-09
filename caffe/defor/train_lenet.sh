#!/usr/bin/env sh
set -e

./../build/tools/caffe train --solver=model_prototxt/lenet_solver.prototxt $@
