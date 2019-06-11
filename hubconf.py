dependencies = ['torch', 'torchvision']

from model import cifar_srm_resnet32
from sotabench.image_classification import cifar10

cifar10.benchmark(model=cifar_srm_resnet32(pretrained=True))
