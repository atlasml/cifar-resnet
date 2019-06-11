dependencies = ['torch', 'torchvision']

from model import cifar_srm_resnet32
from sotabench.vision.image_classification.cifar10 import evaluate_cifar10

evaluate_cifar10(model=cifar_srm_resnet32(pretrained=True))
