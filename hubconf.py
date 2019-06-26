from model import cifar_srm_resnet32
from sotabench.image_classification import CIFAR10

CIFAR10.benchmark(
    model=cifar_srm_resnet32(pretrained=True),
    paper_model_name='SRM ResNet 32',
    paper_arxiv_id='1903.10829',
    paper_pwc_id='srm-a-style-based-recalibration-module-for'
)
