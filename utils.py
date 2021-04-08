from libs.mia.wrappers import *


def minibatch_loader(minibatch, minibatch_size, drop_last=True):
  return DataLoader(minibatch, batch_size=minibatch_size, drop_last=drop_last)


def _numpy_to_dataloader(X, y=None, *args, **kwargs):
  X = torch.from_numpy(X)
  if y is not None:
    y = torch.from_numpy(y)
  return _torch_to_dataloader(X, y, *args, **kwargs)


def _torch_to_dataloader(X, y=None, *args, **kwargs):
  tensors = [X]
  if y is not None:
    tensors.append(y)
  dataset = TensorDataset(*tensors)
  return DataLoader(dataset, *args, **kwargs)


def _input_to_dataloader(X, y=None, offset=None, max_examples=None, *args, **kwargs):
  if offset is None:
    offset = 0
  if max_examples is None:
    max_examples = len(X)
  X_slice = X[offset:max_examples]
  y_slice = y[offset:max_examples] if y is not None else None
  if isinstance(X, np.ndarray):
    return _numpy_to_dataloader(X_slice, y_slice, *args, **kwargs)
  elif isinstance(X, torch.Tensor):
    return _torch_to_dataloader(X_slice, y_slice, *args, **kwargs)
  else:
    raise NotImplementedError()


class DPTorchWrapper(TorchWrapper):

  def fit(
      self,
      X,
      y=None,
      batch_size=32,
      epochs=20,
      shuffle=True,
      validation_split=None,
      validation_data=None,
      verbose=False,
      minibatch_size=1
  ):
    """
    Fit a torch classifier.
    :param X: Dataset
    :type X: ``numpy.ndarray`` or ``torch.Tensor``.
    :param y: Labels
    :param batch_size: Batch size
    :param epochs: Number of epochs to run the training
    :param shuffle: Whether to shuffle the dataset
    :param validation_split: Ratio of data to use for training. E.g., 0.7
    :param validation_data: If ``validation_split`` is not specified,
            the explicit validation dataset.
    :param verbose: Whether to output the progress report.
    TODO: Add custom metrics.
    """
    max_train_examples = None
    val_offset = None

    if validation_split is not None:
      max_train_examples = int((1 - validation_split) * len(X))
      val_offset = int(validation_split * len(X))
      validation_data = (X, y)

    train_loader = _input_to_dataloader(
      X,
      y,
      batch_size=batch_size,
      shuffle=shuffle,
      max_examples=max_train_examples,
    )

    phases = ["train"]
    if validation_data is not None:
      if isinstance(validation_data, tuple):
        self.val_loader_ = val_loader = _input_to_dataloader(
          *validation_data,
          batch_size=batch_size,
          shuffle=False,
          offset=val_offset
        )
      else:
        self.val_loader_ = val_loader = _input_to_dataloader(
          validation_data.data,
          validation_data.targets,
          batch_size=batch_size,
          shuffle=False,
          offset=val_offset
        )
      phases.append("val")
      best_val_loss = 0.0

    since = time.time()

    for epoch in range(epochs):
      if verbose:
        print("Epoch %d/%d" % (epoch + 1, epochs))

      # Each epoch has a training and validation phase.
      for phase in phases:
        if phase == "train":
          self.optimizer_ = self.lr_scheduler(self.optimizer_, epoch)
          data_iter = train_loader
        else:
          data_iter = val_loader

        if verbose and phase == "train":
          prog_bar = tqdm.tqdm(total=len(data_iter.dataset))

        epoch_dataset_size = 0
        running_loss = 0.0
        running_num_correct_preds = 0

        # Run through the data in minibatches.
        for data in data_iter:
          batch_loss, batch_size, num_correct_preds = self.fit_step(
            data, minibatch_size=minibatch_size, phase=phase
          )

          # Batch statistics.
          epoch_dataset_size += batch_size
          running_loss += batch_loss
          running_num_correct_preds += num_correct_preds

          if verbose and phase == "train":
            batch_stats_str = TorchWrapper.STATS_FMT.format(
              phase,
              batch_loss / batch_size,
              num_correct_preds / batch_size,
            )
            prog_bar.set_description(batch_stats_str)
            prog_bar.update(batch_size)

        # Epoch statistics.
        epoch_loss = running_loss / epoch_dataset_size
        epoch_acc = running_num_correct_preds / epoch_dataset_size

        if verbose:
          prog_bar.close()
          epoch_stats_str = TorchWrapper.STATS_FMT.format(
            phase, epoch_loss, epoch_acc
          )
          print(epoch_stats_str)

        # Determine if model is the best.
        if phase == "val" and self.serializer is not None:
          if epoch_loss < best_loss:
            best_loss = epoch_loss
            # TODO: Needs more work. Check the temperature
            #       scaling class. Neural Network should change.
            self.serializer.save(TorchWrapper.BEST_MODEL_ID)
            if verbose:
              print("New best accuracy: %.4f" % best_loss)

    time_elapsed = time.time() - since
    print(
      "Training complete in {:.0f}m {:.0f}s".format(
        time_elapsed // 60, time_elapsed % 60
      )
    )

    # Load the best model.
    if "val" in phases and self.serializer is not None:
      model_wrapper = self.serializer.load(TorchWrapper.BEST_MODEL_ID)
      self.module_ = model_wrapper.module_
      self.val_loader_ = model_wrapper.val_loader_

    return self.module_

  def fit_step(self, batch, minibatch_size=1, phase="train"):
    """
    Run a single training step.
    :param batch: A tuple of numpy batch examples and labels
    :param phase: Phase. One of ['train', 'val']. If in val, does not
                  update the model parameters.
    """
    self.module_.train(phase == "train")
    batch_size = len(batch[0])
    batch_loss = 0

    # Zero the parameter gradients.
    self.optimizer_.zero_grad()

    all_preds = []

    for inputs, labels in minibatch_loader(TensorDataset(*batch), minibatch_size):
      # Wrap them in Variable
      if self.enable_cuda:
        inputs, labels = Variable(inputs.cuda()).float(), Variable(labels.cuda()).long()
      else:
        inputs, labels = Variable(inputs).float(), Variable(labels).long()

      self.optimizer_.zero_minibatch_grad()

      # Forward pass.
      outputs = self.module_(inputs).view(minibatch_size, -1)
      preds = torch.argmax(outputs.data, 1)
      all_preds.append(preds)
      # Calculating the loss.
      loss = self.criterion_(outputs, labels)
      batch_loss += loss.item()

      # Backward + optimize only if in training phase.
      if phase == "train":
        loss.backward()
        self.optimizer_.minibatch_step()

    if phase == "train":
      self.optimizer_.step()

    # Batch statistics.
    if self.enable_cuda:
      all_preds = torch.tensor(all_preds).cuda()
      num_correct_preds = float(torch.sum(all_preds == batch[1].data.cuda()))
    else:
      all_preds = torch.tensor(all_preds)
      num_correct_preds = float(torch.sum(all_preds == batch[1].data))

    return batch_loss, batch_size, num_correct_preds

  def predict_proba(self, X, batch_size=32):
    """Get the confidence vector for an evaluation of a trained model.

    :param X: Data
    :param batch_size: Batch size

    TODO: Fix in case this is not one-hot.

    """
    data_iter = _input_to_dataloader(X, batch_size=batch_size)

    batch_outputs = []
    for batch in data_iter:
      x = batch[0]
      if self.enable_cuda:
        x = x.cuda().float()
      outputs = self.module_(x).data.cpu()
      batch_outputs.append(outputs)
    return np.vstack(batch_outputs)

  def predict(self, X, batch_size=32):
    """Get the confidence vector for an evaluation of a trained model.

    :param X: Data
    :param batch_size: Batch size
    """
    return np.argmax(self.predict_proba(X, batch_size), axis=1)
