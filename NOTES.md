
# Dev Notes（笔记）/ Planning （策划）

Notes for myself, for planning development and keeping track, what to do next.

## Keeping Track

So I don't forget some details.

### Backbone related
Resnet backbone extracts a (1, 25088) == (7, 7, 512) vector before avgpool
and pools to (1, 512).

### Current plan for connection Backbone-CTM

* utilize layer4 (7, 7, 512) vector as input for CTM and find a mask onto a
7x7 feature vector (**shouldn't work well, too small**)

* use the layer3 output (14, 14, 256) instead (**should work better**)

### Concentrator

* using BasicBlock results in {input width - 4 = output width} (e.g. 14 -> 10)

### Projector

* see Concentrator point 1 (e.g. 10 -> 6)

### Reshaper

* reshape to the output dims of projector from input size of the concentrator/output size of the backbone

## Research

### PyTorch-related

* find out, whether a saved tensor retains autograd history or not
  - it does not, so partial training is impossible

* find and type down that formula for convolutions and input -> output sizes
  - (W - K + 2P)/S + 1 where W = input width, K =  kernel width, P = padding amount, S = stride

## Implementation Plan / Roadmap


### TODO (必须)

* work on config file, implement flags for CTM components
* implement network structure for
    - Concentrator
    - Projector
    - Reshaper
* implement full training strategy
* implement GPU/CPU transport to leverage available GPUs
* add terminal/cmd execution mode

### Should TODO （应该）

* implement a kind of logging
* look for more flexible ways to define network components (config file)

## Testing

Testing-related notes.

### General Tests（必须）

* add different metrics

### Datasets（必须）

* test on miniImagenet
* test on Omniglot

### Potential Testing TODO （可能）

* test different approaches/architectures for the components

