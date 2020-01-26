from neuronpp.cells.cell import Cell

from populations.population import Population


class MotorPopuation(Population):
    def _create_single_cell(self) -> Cell:
        cell = Cell("output")
        cell.make_sec("soma", diam=5, l=5, nseg=1)
        cell.insert('pas')
        cell.insert('hh')
        cell.make_spike_detector()
        return cell

    def _connect_sigle_cell(self, source_section, cell, loc, weight=1, **kwargs) -> list:
        syns = cell.make_sypanses(source=source_section, weight=weight, mod_name="ExpSyn", sec="soma", source_loc=loc,
                                  target_loc=0.5, threshold=10, e=40, tau=3, delay=3)
        return syns
