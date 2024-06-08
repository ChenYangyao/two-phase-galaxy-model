import two_phase_galaxy_model as tpgm
from two_phase_galaxy_model import samples
from two_phase_galaxy_model.recipes import subclouds, subhalos, star_cluster_model, galaxy_model
import numpy as np
import pytest
from pyhipp import plot
from typing import Sequence, Mapping
from pyhipp.core import DataTable

@pytest.fixture
def raw_merger_tree() -> dict:
    merger_tree = {
        'id': [10, 11, 12, 13, 14, 
                15, 16, 17, 18,
                        19, 20],
        'leaf_id': [14, 14, 14, 14, 14,
                        18, 18, 18, 18,
                                20, 20],
        'last_pro_id': [20, 14, 14, 14, 14,
                            20, 20, 18, 18,
                                    20, 20],
        'snap': [99, 97, 80, 70, 55,
                    97, 80, 70, 55,
                            70, 55],
        'z': [0.0, 1.0, 2.0, 3.0, 4.0,
                1.0, 2.0, 3.0, 4.0,
                            3.0, 4.0],
        'is_cent': [True, True,  True, False, True,
                        False, True, True,  True,
                                    True,  True],
        'm_h': [10**2.5, 10**2.0, 10**1.5, 0.0,     10**0.5,
                        0.0,     10**1.0, 10**1.0, 10**0.5,
                                        10**0.5, 10**-0.5],
        'v_max': [10**2.8, 10**2.2, 10**1.7, 10**1.6, 10**1.5,
                        10**1.9, 10**1.7, 10**1.6, 10**1.5,
                                            10**1.5, 10**1.0],
    }
    return merger_tree



def test_raw_merger_tree(raw_merger_tree: dict):
    for k, v in raw_merger_tree.items():
        assert isinstance(k, str)
        assert isinstance(v, Sequence)
        
@pytest.fixture
def merger_tree(raw_merger_tree: dict) -> DataTable:
    merger_tree = tpgm.samples.make_group_tree_from_dict(raw_merger_tree)
    return merger_tree


def test_merger_tree(merger_tree: DataTable):
    assert isinstance(merger_tree, Mapping)
    
    
@pytest.fixture
def model() -> tpgm.model.GroupTreeModel:
    model = tpgm.model.GroupTreeModel()
    model.update_parameters({
        'subclouds_in_group_tree.samplers.beta_m': -2.1,
    })
    model.set_up()
    return model

    
def test_model_run(model: tpgm.model.GroupTreeModel, merger_tree: DataTable):
    out = model(merger_tree)
    assert out.grptr.size > 0
    assert out.sc_set.size > 0