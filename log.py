import sys


class Logger:
  STD_files = [sys.stdout, sys.stderr]

  def __init__(self, log_files):
    self.log_files = []
    for file in log_files:
      if isinstance(file, str):
        self.log_files.append(open(file, 'w+'))
      else:
        self.log_files.append(file)  # assuming file is a valid open file

  def __del__(self):
    self.close()

  def close(self):
    for file in self.log_files:
      if file not in Logger.STD_files:
        file.close()

  def log(self, *message, module=''):
    for file in self.log_files:
      print(f'[{module}]', *message, file=file)

  def log_waring(self, *message, module=''):
    self.log(*message, module)
    print(f'[{module}]', *message, file=sys.stderr)