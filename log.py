import sys
from datetime import datetime

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
    time = datetime.now()
    time = str(time.strftime("%d/%m/%Y,%H:%M:%S"))
    for file in self.log_files:
      print(f'[{time}] [{module}]', *message, file=file)

  def log_waring(self, *message, module=''):
    time = datetime.now()
    time = str(time.strftime("%d/%m/%Y,%H:%M:%S"))
    self.log(*message, module)
    print(f'[{time}] [{module}]', *message, file=sys.stderr)