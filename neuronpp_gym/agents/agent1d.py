from neuronpp_gym.core.agent_core import AgentCore


class Agent1D(AgentCore):
    def build(self, input_size: int):
        """
        Build Agent for 1 Dimensional input.

        Before Agent step() you need to call:
          1. agent.build()
          2. agent.init()
        """
        if self._built:
            raise RuntimeError("The Agent have been already built.")
        if not isinstance(input_size, int):
            raise ValueError("input_size can only be of type int.")
        self._build(input_shape=(input_size,), input_size=input_size, input_cell_num=input_size)

    def _make_observation(self, observation, poisson=False, stepsize=None):
        """
        Make 1D input observation

        :param observation:
            1 dim array of numbers
        :return:
        """
        if self.input_size != observation.size:
            raise RuntimeError("Observation must be of same size as self.input_size, which is "
                               "a product of input_shape.")

        input_syns = [s for c in self.input_cells for s in c.syns]
        self._make_single_observation(observation=observation, syns=input_syns, poisson=poisson,
                                      stepsize=stepsize)