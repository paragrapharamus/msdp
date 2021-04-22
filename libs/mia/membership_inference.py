import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split

from datasets.dataset_util import *
from dp.dp_optim import DPSGD
from libs.mia.estimators import ShadowModelBundle, AttackModelBundle, prepare_attack_data
from libs.mia.wrappers import TorchWrapper, DPTorchWrapper
from models import *


def get_target_model(args):
  model_cls = Cifar10Net
  optimizer_cls = optim.SGD
  criterion_cls = nn.CrossEntropyLoss
  optimizer_params = {
    'lr': args.lr,
    'momentum': args.momentum,
    'weight_decay': args.weight_decay,
  }
  return TorchWrapper(
    model_cls,
    criterion_cls,
    optimizer_cls,
    optimizer_params=optimizer_params,
    lr_scheduler=None,
    enable_cuda=True,
    serializer=None,
  )


def get_shadow_model():
  model_cls = Cifar10Net
  optimizer_cls = torch.optim.Adam
  criterion_cls = nn.CrossEntropyLoss

  return TorchWrapper(
    model_cls,
    criterion_cls,
    optimizer_cls,
    lr_scheduler=None,
    enable_cuda=True,
    serializer=None,
  )


def get_attack_model(enable_cuda=True):
  return TorchWrapper(
    AttackModel,
    nn.BCELoss,
    optim.Adam,
    module_params={'num_classes': 10},
    lr_scheduler=None,
    enable_cuda=enable_cuda,
    serializer=None
  )


def membership_inference_attack(args,
                                num_classes,
                                shadow_dataset_size,
                                attack_test_dataset_size,
                                num_shadows,
                                attack_epochs):
  RAND_SEED = args.seed

  use_cuda = not args.no_cuda and torch.cuda.is_available()
  torch.manual_seed(args.seed)
  train_kwargs = {"batch_size": args.batch_size}
  test_kwargs = {"batch_size": args.test_batch_size}
  if use_cuda:
    cuda_kwargs = {"num_workers": 1, "pin_memory": True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

  train_dataset, _, test_dataset = get_cifar10_dataset()

  X_train, y_train = np.moveaxis(train_dataset.data, -1, 1), np.array(train_dataset.targets)
  X_test, y_test = np.moveaxis(test_dataset.data, -1, 1), np.array(test_dataset.targets)

  target_model = get_target_model(args)
  if args.trained_model_path is not None:
    print('Loading the victim model')
    target_model.module_.load_state_dict(torch.load(args.trained_model_path))
  else:
    print('Training the victim model')
    target_model.fit(X=X_train,
                     y=y_train,
                     batch_size=args.batch_size,
                     epochs=args.epochs,
                     shuffle=True,
                     validation_split=0.1,
                     verbose=True,
                     )
    if args.save_model:
      torch.save(target_model.module_.statde_dict(), args.save_model_path)

  # Train the shadow models.
  smb = ShadowModelBundle(
    get_shadow_model,
    shadow_dataset_size=shadow_dataset_size,
    num_models=num_shadows,
    seed=RAND_SEED
  )

  # We assume that attacker's data were not seen in target's training.
  attacker_X_train, attacker_X_test, attacker_y_train, attacker_y_test = train_test_split(
    X_test, y_test, test_size=0.1
  )

  print('Attacker dataset shapes')
  print(attacker_X_train.shape, attacker_X_test.shape)

  print("Training the shadow models...")
  X_shadow, y_shadow = smb.fit_transform(
    attacker_X_train,
    attacker_y_train,
    fit_kwargs=dict(
      epochs=args.epochs,
      verbose=True,
      validation_data=(attacker_X_test, attacker_y_test),
    ),
  )

  # ShadowModelBundle returns data in the format suitable for the AttackModelBundle.
  amb = AttackModelBundle(get_attack_model, num_classes=num_classes, class_one_hot_coded=False)

  # Fit the attack models.
  print("Training the attack models...")
  amb.fit(
    X_shadow, y_shadow, fit_kwargs=dict(epochs=attack_epochs, verbose=True)
  )

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

  print(f"MIA accuracy: ", attack_accuracy)
