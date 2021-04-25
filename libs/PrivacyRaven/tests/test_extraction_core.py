import pytest
import torch

import privacyraven as pr
from libs.PrivacyRaven.src.privacyraven.extraction.core import ModelExtractionAttack
from libs.PrivacyRaven.src.privacyraven.models.four_layer import FourLayerClassifier
from libs.PrivacyRaven.src.privacyraven.models.victim import train_four_layer_mnist_victim
from libs.PrivacyRaven.src.privacyraven.utils.data import get_emnist_data
from libs.PrivacyRaven.src.privacyraven.utils.query import get_target


def test_extraction():
    """End-to-end test of a model extraction attack"""

    # Create a query function for a target PyTorch Lightning model
    model = train_four_layer_mnist_victim(gpus=torch.cuda.device_count())

    def query_mnist(input_data):
        # PrivacyRaven provides built-in query functions
        return get_target(model, input_data, (1, 28, 28, 1))

    # Obtain seed (or public) data to be used in extraction
    emnist_train, emnist_test = get_emnist_data()

    # Run a model extraction attack
    attack = ModelExtractionAttack(
        query=query_mnist,
        query_limit=100,
        victim_input_shape=(1, 28, 28, 1),  # EMNIST data point shape
        victim_output_targets=10,
        substitute_input_shape=(3, 1, 28, 28),
        synthesizer="copycat",
        substitute_model_arch=FourLayerClassifier,  # 28*28: image size
        substitute_input_size=784,
        seed_data_train=emnist_train,
        seed_data_test=emnist_test,
        gpus=0,
    )
