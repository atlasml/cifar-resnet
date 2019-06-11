dependencies = ['torch', 'torchvision']

from model import cifar_srm_resnet32
import sotabench.image_classification.cifar10

cifar10.benchmark(model=cifar_srm_resnet32(pretrained=True))
