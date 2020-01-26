from abc import ABC
from neuronpp.cells.cell import Cell
from neuronpp.cells.ebner2019_ach_da_cell import Ebner2019AChDACell
from populations.population import Population


class ExcitatoryPopulation(Population, ABC):
    def _create_single_cell(self) -> Cell:
        name = "input_cell%s" % self.cell_counter
        cell = Ebner2019AChDACell(name, compile_paths="agents/utils/mods/ebner2019 agents/utils/mods/4p_ach_da_syns")
        cell.make_sec("soma", diam=20, l=20, nseg=10)
        cell.make_sec("dend", diam=8, l=500, nseg=100)
        cell.connect_secs(source="dend", target="soma", source_loc=0, target_loc=1)
        return cell