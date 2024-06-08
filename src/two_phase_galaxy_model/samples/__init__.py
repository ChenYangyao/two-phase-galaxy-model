from . import loaders
from pyhipp.astro.cosmology.model import predefined as predefined_cosms
import numpy as np

default_cosm = predefined_cosms['tng']


def make_group_tree_from_dict(
        data: dict[str, np.ndarray], *,
        cosm=default_cosm,
        copy=True):
    '''
    Convert a dictionary of arrays to a GroupTree object to pass into 
    the pipeline of the star cluster model.
    
    Subhalos should exactly form a DFS ordered merger tree.
    
    Each array in `data` contains a given property of subhalos.
    
    @data: must contain items of 
        - id:
            Subhalo index. Should be unique within the `data`, contiguous
            within each subhalo merger tree.
        - leaf_id: 
            The `id` of the main leaf progenitor.
        - last_pro_id: 
            The `id` of the last progenitor in the subhalo merger tree.
        - snap: 
            Snapshot number.
        - z: 
            Redshift. 
        - is_cent: 
            A boolean value indicating whether the subhalo is central. 
        - m_h [10^10 Msun/h]: 
            Halo mass (defined by an overdensity of 200 times critical density).
            For a satellite, can be arbitrary value.
        - v_max [km/s]: 
            Maximum circular velocity.
    '''

    return loaders.GroupTreeLoader(cosm).from_dict(data, copy=copy)
