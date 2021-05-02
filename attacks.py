from copy import deepcopy
from typing import Tuple, Type, Optional, Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from art.attacks.extraction import KnockoffNets
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from art.estimators.classification.pytorch import PyTorchClassifier
from libs.PrivacyRaven.src.privacyraven.extraction.core import ModelExtractionAttack
from libs.PrivacyRaven.src.privacyraven.models.victim import convert_to_inference
from libs.PrivacyRaven.src.privacyraven.utils.query import get_target
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split, Dataset

from datasets.dataset_util import merge_datasets
from libs.mia.estimators import ShadowModelBundle, AttackModelBundle, prepare_attack_data
from libs.mia.wrappers import TorchWrapper


def log(msg, logger=None):
  if logger:
    logger.log(msg)
  else:
    print(msg)


def model_extraction_knockoffnets(model: pl.LightningModule,
                                  attack_model_cls: Type[pl.LightningModule],
                                  loss: nn.modules.loss._Loss,
                                  optimizer_class: Type[optim.Optimizer],
                                  input_shape: Tuple[int, ...],
                                  num_classes: int,
                                  batch_size: int,
                                  epochs: int,
                                  query_limit: int,
                                  test_dataset: Dataset,
                                  logger=None
                                  ):
  log("KnockOffNets Model extraction attack has started...", logger)
  if len(input_shape) == 4:
    input_shape = input_shape[1:]

  victim_model = PyTorchClassifier(model=model,
                                   loss=loss,
                                   optimizer=optimizer_class(model.parameters()),
                                   input_shape=input_shape,
                                   nb_classes=num_classes)

  attack = KnockoffNets(classifier=victim_model,
                        batch_size_fit=batch_size,
                        nb_epochs=epochs,
                        nb_stolen=query_limit
                        )

  indices = np.random.permutation(len(test_dataset))
  X_test, y_test = np.moveaxis(test_dataset.data, -1, 1), np.array(test_dataset.targets)
  X_steal = X_test[indices[:query_limit]].astype(np.float32)
  y_steal = y_test[indices[:query_limit]].astype(np.float32)
  X_test = X_test[indices[query_limit:]].astype(np.float32)
  y_test = y_test[indices[query_limit:]].astype(np.float32)

  extracted_model = PyTorchClassifier(model=attack_model_cls(),
                                      loss=loss,
                                      optimizer=optimizer_class(model.parameters()),
                                      input_shape=input_shape,
                                      nb_classes=num_classes
                                      )
  log("Stealing the model...", logger)
  extracted_model = attack.extract(X_steal, y_steal, thieved_classifier=extracted_model)
  log("Testing...", logger)
  victim_model_preds = np.argmax(victim_model.predict(X_test), axis=1)
  extracted_model_preds = np.argmax(extracted_model.predict(X_test), axis=1)
  acc = np.sum(victim_model_preds == extracted_model_preds) / len(y_test)
  log(f"Extraction fidelity: {100 * acc:.2f}%", logger)
  return acc


def model_extraction(model: pl.LightningModule,
                     train_dataset: Dataset,
                     test_dataset: Dataset,
                     query_limit: int,
                     victim_input_shape: Tuple[int, ...],
                     num_classes: int,
                     attacker_input_shape: Tuple[int, ...],
                     synthesizer_name: str,
                     substitute_architecture: Type[pl.LightningModule],
                     max_epochs: int,
                     logger=None
                     ):
  def truncate(ds: Dataset, size: int):
    idxs = np.random.permutation(len(ds.data))[:size]
    ds.data = ds.data[idxs]
    ds.targets = np.array(ds.targets)[idxs]
    return ds

  log('Model extraction has started...', logger)
  # The victim model
  model = convert_to_inference(model)

  # Obtain seed (or public) data to be used in extraction
  train_dataset = truncate(deepcopy(train_dataset), query_limit // 2)
  test_dataset = truncate(deepcopy(test_dataset), query_limit // 2)
  dataset = merge_datasets(train_dataset, test_dataset)

  split_ratio = 0.1
  train_data, test_data = random_split(dataset,
                                       [int((1 - split_ratio) * len(dataset)),
                                        int(split_ratio * len(dataset))])
  attack_train_data = [train_data.dataset[i] for i in train_data.indices]
  attack_test_data = [test_data.dataset[i] for i in test_data.indices]

  # Create a query function for a target PyTorch Lightning model
  def query_fn(input_data):
    return get_target(model, input_data, victim_input_shape)

  # Run a model extraction attack
  attack = ModelExtractionAttack(
    query_fn,
    query_limit,
    victim_input_shape,
    num_classes,
    attacker_input_shape,
    synthesizer_name,
    substitute_architecture,
    np.prod(attacker_input_shape),
    attack_train_data,
    attack_test_data,
    max_epochs=max_epochs,
    trainer_args={'progress_bar_refresh_rate': 0,
                  'weights_summary': None}
  )

  # extraction_rate = _calc_extraction_rate(model, attack.substitute_model)

  log(f"Model extraction fidelity: {attack.label_agreement * 100:.2f}", logger)
  # log(attack.__dict__)
  return attack.label_agreement


def membership_inference_black_box(model: pl.LightningModule,
                                   loss: nn.modules.loss._Loss,
                                   optimizer_class: Type[optim.Optimizer],
                                   train_dataset: Dataset,
                                   test_dataset: Dataset,
                                   input_shape: Tuple[int, ...],
                                   num_classes: int,
                                   attack_train_size: int,
                                   attack_test_size: int,
                                   attack_model_type: Optional[str] = 'rf',
                                   attack_model: Optional[Any] = None,
                                   logger=None
                                   ):
  log('Membership Inference attack...', logger)
  if len(input_shape) == 4:
    input_shape = input_shape[1:]

  art_model = PyTorchClassifier(model=model,
                                loss=loss,
                                optimizer=optimizer_class(model.parameters()),
                                input_shape=input_shape,
                                nb_classes=num_classes)

  bb_atack = MembershipInferenceBlackBox(art_model,
                                         attack_model_type=attack_model_type,
                                         attack_model=attack_model
                                         )
  train_dataset = deepcopy(train_dataset)
  test_dataset = deepcopy(test_dataset)

  x_train, y_train = np.moveaxis(train_dataset.data, -1, 1), np.array(train_dataset.targets)
  x_test, y_test = np.moveaxis(test_dataset.data, -1, 1), np.array(test_dataset.targets)

  # Train the attack model
  log('Training the attack model...', logger)
  bb_atack.fit(x_train[:attack_train_size].astype(np.float32), y_train[:attack_train_size],
               x_test[:attack_test_size].astype(np.float32), y_test[:attack_test_size])

  # Infer on member and non-member data
  log('Attack the target model model...', logger)
  s, e = attack_train_size, min(2 * attack_train_size, len(x_train))
  train_members = bb_atack.infer(x_train.astype(np.float32)[s:e], y_train[s:e])
  s, e = attack_test_size, min(2 * attack_test_size, len(x_test))
  test_members = bb_atack.infer(x_test.astype(np.float32)[s:e], y_test[s:e])

  # Check the accuracy of the attack
  train_acc_attack = np.sum(train_members) / len(train_members)
  test_acc_attack = 1 - (np.sum(test_members) / len(test_members))

  attack_acc = (train_acc_attack * len(train_members) + test_acc_attack * len(test_members)) / \
               (len(train_members) + len(test_members))
  log(f"MIA accuracy on training data {train_acc_attack}", logger)
  log(f"MIA accuracy on test data {test_acc_attack}", logger)

  p, r = _calc_precision_recall(np.concatenate((train_members, test_members)),
                                np.concatenate((np.ones(len(train_members)), np.zeros(len(test_members)))))

  log(f"MIA accuracy: {attack_acc:.3f}, precision: {p:.3f}, recall: {r:.3f}", logger)
  return attack_acc, p, r


def membership_inference_mia(model: pl.LightningModule,
                             model_cls: Type[pl.LightningModule],
                             attack_model_cls: Type[pl.LightningModule],
                             num_classes: int,
                             train_dataset: Dataset,
                             test_dataset: Dataset,
                             shadow_epochs: int,
                             shadow_dataset_size: int,
                             attack_test_dataset_size: int,
                             num_shadows: int,
                             attack_epochs: int,
                             seed: int,
                             use_cuda: bool = True,
                             logger=None
                             ):
  def get_target_model():
    optimizer_cls = optim.Adam
    criterion_cls = nn.CrossEntropyLoss
    return TorchWrapper(
      model_cls,
      criterion_cls,
      optimizer_cls,
      enable_cuda=use_cuda
    )

  def get_attack_model():
    return TorchWrapper(
      attack_model_cls,
      nn.BCELoss,
      optim.Adam,
      module_params={'num_classes': num_classes},
      enable_cuda=use_cuda,
    )

  def get_shadow_model():
    optimizer_cls = torch.optim.Adam
    criterion_cls = nn.CrossEntropyLoss

    return TorchWrapper(
      model_cls,
      criterion_cls,
      optimizer_cls,
      enable_cuda=use_cuda,
    )

  X_train, y_train = np.moveaxis(train_dataset.data, -1, 1), np.array(train_dataset.targets)
  X_test, y_test = np.moveaxis(test_dataset.data, -1, 1), np.array(test_dataset.targets)

  target_model = get_target_model()
  log('Loading the victim model', logger)
  target_model.module_.load_state_dict(model.state_dict())

  # Building the shadows
  smb = ShadowModelBundle(
    get_shadow_model,
    shadow_dataset_size=shadow_dataset_size,
    num_models=num_shadows,
    seed=seed
  )

  # We assume that attacker's data were not seen in target's training.
  (attacker_X_train, attacker_X_test,
   attacker_y_train, attacker_y_test) = train_test_split(X_test, y_test, test_size=0.1)

  log(f'Attacker dataset shapes -> {attacker_X_train.shape, attacker_X_test.shape}', logger)

  log("Training the shadow models...", logger)
  X_shadow, y_shadow = smb.fit_transform(
    attacker_X_train,
    attacker_y_train,
    fit_kwargs=dict(
      epochs=shadow_epochs,
      verbose=True,
      validation_data=(attacker_X_test, attacker_y_test),
    ),
  )

  # ShadowModelBundle returns data in the format suitable for the AttackModelBundle.
  amb = AttackModelBundle(get_attack_model,
                          num_classes=num_classes,
                          class_one_hot_coded=False)

  # Fit the attack models.
  log("Training the attack models...", logger)
  amb.fit(X_shadow, y_shadow,
          fit_kwargs=dict(epochs=attack_epochs, verbose=True))

  # Test the success of the attack.
  # Prepare examples that were in the training, and out of the training.
  data_in = X_train[:attack_test_dataset_size], y_train[:attack_test_dataset_size]
  data_out = X_test[:attack_test_dataset_size], y_test[:attack_test_dataset_size]

  # Compile them into the expected format for the AttackModelBundle.
  attack_test_data, real_membership_labels = prepare_attack_data(
    target_model, data_in, data_out
  )

  # Compute the attack accuracy.
  attack_guesses = amb.predict(attack_test_data)
  attack_accuracy = np.mean(attack_guesses == real_membership_labels)

  log(f"MIA accuracy: {attack_accuracy}", logger)
  return attack_accuracy


def _calc_precision_recall(predicted, actual, positive_value=1):
  score = 0  # both predicted and actual are positive
  num_positive_predicted = 0  # predicted positive
  num_positive_actual = 0  # actual positive
  for i in range(len(predicted)):
    if predicted[i] == positive_value:
      num_positive_predicted += 1
    if actual[i] == positive_value:
      num_positive_actual += 1
    if predicted[i] == actual[i]:
      if predicted[i] == positive_value:
        score += 1

  if num_positive_predicted == 0:
    precision = 1
  else:
    precision = score / num_positive_predicted  # the fraction of predicted “Yes” responses that are correct
  if num_positive_actual == 0:
    recall = 1
  else:
    recall = score / num_positive_actual  # the fraction of “Yes” responses that are predicted correctly

  return precision, recall


def _calc_extraction_rate(victim_model, extracted_model):
  victim_model_params = list(victim_model.parameters())
  extracted_model_params = list(extracted_model.parameters())

  diffs = []
  for i in range(len(victim_model_params)):
    diffs.append(
      torch.linalg.norm(extracted_model_params[i] - victim_model_params[i]) / torch.linalg.norm(victim_model_params[i]))

  diffs = sum(diffs) / len(diffs)
  return 1 - min(diffs, 1)
