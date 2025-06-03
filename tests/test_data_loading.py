import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import load_datasets

def test_load_datasets():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets')
    data_dir = os.path.abspath(data_dir)
    data = load_datasets(data_dir)
    assert 'text' in data.columns
    assert 'class' in data.columns
    assert len(data[data['class'] == 0]) == 23481
    assert len(data[data['class'] == 1]) == 21417
