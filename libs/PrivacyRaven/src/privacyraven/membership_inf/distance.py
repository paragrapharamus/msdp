import torch
from art.attacks.evasion import BoundaryAttack, HopSkipJump
from art.estimators.classification import BlackBoxClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from libs.PrivacyRaven.src.privacyraven.extraction.synthesis import hopskipjump
from libs.PrivacyRaven.src.privacyraven.utils.model_creation import NewDataset
from libs.PrivacyRaven.src.privacyraven.utils.query import reshape_input

dist = dict()
