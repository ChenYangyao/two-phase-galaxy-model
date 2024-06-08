from __future__ import annotations
from typing import NamedTuple
from pyhipp.core import DataTable
from .abc import Model
from .subhalos import GroupTree
from .subclouds import SubcloudSet
from .preproc import PreprocGroupTree
from .galaxy_model import GalaxiesInGroupTree
from .star_cluster_model import SubcloudsInGroupTree


class GroupTreeModel(Model):

    class Result(NamedTuple):
        grptr: GroupTree
        sc_set: SubcloudSet

    def __init__(self, **kw):

        super().__init__(**kw)

        self.preproc_group_tree = PreprocGroupTree()
        self.galaxies_in_group_tree = GalaxiesInGroupTree()
        self.subclouds_in_group_tree = SubcloudsInGroupTree()

    def __call__(self, grptr_raw: DataTable):
        log = self.ctx.log

        grptr = self.preproc_group_tree(grptr_raw)

        self.galaxies_in_group_tree(grptr)

        sc_set = self.subclouds_in_group_tree(grptr)

        return self.Result(grptr, sc_set)


class GroupTreeBatchGalaxy(Model):

    class Result(NamedTuple):
        grptrs: list[GroupTree]

    def __init__(self, **kw) -> None:

        super().__init__(**kw)

        self.preproc_group_tree = PreprocGroupTree()
        self.galaxies_in_group_tree = GalaxiesInGroupTree()

    def __call__(
            self, grptrs_raw: list[DataTable]):
        preproc = self.preproc_group_tree
        model = self.galaxies_in_group_tree

        grptrs = [preproc(grptr_raw) for grptr_raw in grptrs_raw]
        for grp in grptrs:
            model(grp)

        return self.Result(grptrs)


class GroupTreeBatchSubcloud(Model):

    class Result(NamedTuple):

        grptrs: list[GroupTree]
        sc_sets: list[SubcloudSet]

        def iter(self):
            return zip(self.grptrs, self.sc_sets)

    def __init__(self, **kw) -> None:

        super().__init__(**kw)

        self.subclouds_in_group_tree = SubcloudsInGroupTree()

    def __call__(self, grptrs: list[GroupTree]):

        model = self.subclouds_in_group_tree
        sc_sets = [model(grptr) for grptr in grptrs]

        return self.Result(grptrs, sc_sets)
