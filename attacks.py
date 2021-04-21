import numpy as np
from privacyraven.extraction.core import ModelExtractionAttack
from privacyraven.membership_inf.core import TransferMembershipInferenceAttack
from privacyraven.utils.query import get_target
from privacyraven.models.victim import convert_to_inference


def model_extraction(model,
                     query_limit,
                     victim_input_shape,
                     number_of_targets,
                     attacker_input_shape,
                     synthesizer_name,
                     substitute_architecture,
                     attack_train_data,
                     attack_test_data):
  # The victim model
  model = convert_to_inference(model)

  # Create a query function for a target PyTorch Lightning model
  def query_fn(input_data):
    return get_target(model, input_data, victim_input_shape)

  # Run a model extraction attack
  attack = ModelExtractionAttack(
    query_fn,
    query_limit,
    victim_input_shape,
    number_of_targets,
    attacker_input_shape,
    synthesizer_name,
    substitute_architecture,
    np.prod(attacker_input_shape),
    attack_train_data,
    attack_test_data,
  )
  # print(attack.__dict__)


def membership_inference(model,
                         query_limit,
                         victim_input_shape,
                         number_of_targets,
                         attacker_input_shape,
                         synthesizer_name,
                         substitute_architecture,
                         attack_train_data,
                         attack_test_data):
  # The victim model
  model = convert_to_inference(model)

  # Create a query function for a target PyTorch Lightning model
  def query_fn(input_data):
    return get_target(model, input_data, victim_input_shape)

  attack = TransferMembershipInferenceAttack(
    query_fn,
    query_limit,
    victim_input_shape,
    number_of_targets,
    attacker_input_shape,
    synthesizer_name,
    substitute_architecture,
    np.prod(attacker_input_shape),
    attack_train_data,
    attack_test_data,
  )
  # print(attack.__dict__)
