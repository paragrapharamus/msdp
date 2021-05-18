import sys
from datetime import datetime


class Logger:
  STD_files = [sys.stdout, sys.stderr]

  def __init__(self, log_files, max_buffer_size=0):
    self.log_files = []
    for file in log_files:
      if isinstance(file, str):
        self.log_files.append(open(file, 'w+'))
      else:
        self.log_files.append(file)  # assuming file is a valid open file
    self.buffer = []
    self._MAX_BUFFER_SIZE = max_buffer_size

  def __del__(self):
    self.close()

  def close(self):
    self._empty_buffer()
    for file in self.log_files:
      if file not in Logger.STD_files:
        file.close()

  def log(self, *message, module=''):
    time = datetime.now()
    time = str(time.strftime("%d/%m/%Y,%H:%M:%S"))
    msg = f"[{time}] [{module}] {' '.join(message)}\n"
    self.buffer.append(msg)
    if len(self.buffer) > self._MAX_BUFFER_SIZE:
      self._empty_buffer()

  def _empty_buffer(self):
    if len(self.buffer) > 0:
      content = ''.join(self.buffer)
      for file in self.log_files:
        print(content, file=file, end='')
      self.buffer = []

  def log_waring(self, *message, module=''):
    time = datetime.now()
    time = str(time.strftime("%d/%m/%Y,%H:%M:%S"))
    self.log(*message, module)
    print(f'[{time}] [{module}]', *message, file=sys.stderr)
