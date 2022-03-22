# U2Net version2
this repository is made from  [2020, PR] U2Net, Going Deeper with NEsted U-structure for salient object detection 

See this [paper](https://arxiv.org/abs/2005.09007)

# Checklists
- [ ] how to control logs?
- [ ] ddp model -> need to run initializer of distributed.init()
- [x] config file name is under character
 

## What's news
- use python-hydra to manage configs file
- use weight and bias to optimize hyper-parameters

# Requirements
- cuda: 11.1 version
  - if you have different cuda version will chnage cuda version in requirements.txt before install all packages

## Installation
~~~
pip install -r requirements.txt
~~~

## Gettting Started