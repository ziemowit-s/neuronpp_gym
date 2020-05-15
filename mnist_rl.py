import argparse

import numpy as np

from agents.agent import Kernel
from agents.ebner_agent import EbnerAgent
from mnist_rl_utils import AgentParam, LinApprox, get_cell_weights, get_observation, get_predicted_reward
from mnist_rl_utils import mnist_prepare

# todo move these constants or build a class
AGENT_STEPSIZE = 100
MNIST_LABELS = 3
SKIP_PIXELS = 2
INPUT_CELL_NUM = 9
DT = 0.2
RESTING_POTENTIAL = -70


def build_agent(run_params, in_data_shape, kernel_size, padding, stride):
    agent = EbnerAgent(output_cell_num=run_params.label_num, input_max_hz=800, default_stepsize=run_params.stepsize,
                       ach_tau=2, da_tau=50)
    # agent = EbnerAgentInhb(output_cell_num=MNIST_LABELS, input_max_hz=800, default_stepsize=AGENT_STEPSIZE)
    # agent = EbnerOlfactoryAgent(output_cell_num=MNIST_LABELS, input_max_hz=800, default_stepsize=AGENT_STEPSIZE)
    # agent = EbOlA(output_cell_num=MNIST_LABELS, input_max_hz=800, default_stepsize=AGENT_STEPSIZE)
    # todo warning attention -- the actual input_shape depends on the skip
    # todo why is it shape[1] and shape[2]????????????????????????
    agent.build(input_shape=(in_data_shape[1] // run_params.skip, in_data_shape[2] // run_params.skip),
                x_kernel=Kernel(size=kernel_size, padding=padding, stride=stride),
                y_kernel=Kernel(size=kernel_size, padding=padding, stride=stride))
    agent.init(init_v=run_params.rest_potential, warmup=int(args.display * run_params.stepsize / run_params.dt),
               dt=run_params.dt)
    print("Agent {:s}\n\tinput cells: {:d}\n\tpixels per input: {:d}\n\tlearning with {} RL".format(
        agent.__class__.__name__, agent.input_cell_num, agent.x_kernel.size * agent.y_kernel.size, args.rl))
    return agent


def mnist_rl_run(args):
    x_train, y_train = mnist_prepare(num=MNIST_LABELS)
    # warning  use cv2 downsampling function: in agent build pass shapes _after_ downsampling
    # x_train = x_train[:, ::SKIP_PIXELS, ::SKIP_PIXELS]

    # todo move constants to a special class?
    kernel_size = 5
    padding = 3
    stride = 3
    np.random.seed(19283765)

    # todo insert init values
    run_params = AgentParam(name="EbnerAgent", stepsize=AGENT_STEPSIZE, label_num=MNIST_LABELS,
                            skip=SKIP_PIXELS, input_num=INPUT_CELL_NUM, dt=DT,
                            rest_potential=RESTING_POTENTIAL, out_labels=[0, 1, 2])
    # info build the agent architecture
    agent = build_agent(run_params=run_params, in_data_shape=x_train.shape, kernel_size=kernel_size, padding=padding,
                        stride=stride)
    actor_critic_learn(agent=agent, X=x_train, y=y_train)


def rl_choose_action(agent):
    if np.random.rand() < 0.5:
        return 0
    return 1


def actor_critic_learn(agent, X, y):
    # check the parameter vector
    state = get_cell_weights(cells=agent.output_cells, weight_name="w")
    action_number = 2
    all_actions = np.array([-1, +1])
    net = LinApprox(state_len=state.shape[0], act_no=action_number)
    # phi_w_0 = net.phi_state_action(w, 0)
    # phi_w_1 = net.phi_state_action(w, 1)
    # q_w_0 = net.q_val(state=w, action=0)
    # q_w_1 = net.q_val(state=w, action=1)
    omega = np.array([0.5, 1])
    action_0 = all_actions[0]
    while True:
        # shuffle data
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        for idx in indices:
            r, new_state, predicted, true, small_reward = execute_in_world(X=X, y=y, idx=idx, agent=agent,
                                                                           action=action_0)
            action_1 = draw(y, omega)


def sarsa_lambda_approx(x, a_0, r, y, a_1, net: LinApprox, z):
    alpha = 0.1
    gamma = 0.9
    lambda_par = 1.0
    delta = r + gamma * net.q_val(y, a_1) - net.q_val(x, a_0)
    net.update_eligibility(state=x, action=a_0, gamma=gamma, lambda_par=lambda_par)
    net.update_theta(alpha=alpha, delta=delta)


def draw(y, omega):
    return 0


def execute_in_world(X, y, idx, agent, action):
    old_state = get_cell_weights(cells=agent.output_cells, weight_name='w')
    # todo czy reward_step na pewno tutaj? Czy też w funkcji macierzystej?
    agent.reward_step(reward=action, stepsize=300)
    output = agent.step(observation=get_observation(X, idx), output_type="rate", epsilon=1)
    predicted, reward = get_predicted_reward(output, y[idx])
    new_state = get_cell_weights(cells=agent.output_cells, weight_name="w")
    # diff_state = new_state - old_state
    return -1, new_state, predicted, y[idx], reward


# todo rebuild
def rl_naive(agent, output, x_train, y_train, index, qval, stepsize, tries=100, alpha=0.5, gamma=0.9):
    # todo dwukrotne obliczanie tego samego get_predicted_reward(): tu i na koncu
    _, last_reward = get_predicted_reward(output=output, true=y_train[index])
    agent.reward_step(reward=last_reward, stepsize=stepsize)
    new_index = (index + 1) % y_train.shape[0]
    new_obs = get_observation(x_train_obs=x_train[new_index])
    new_output = agent.step(observation=new_obs, output_type="rate", epsilon=1)
    new_true = y_train[new_index]
    new_predicted, new_reward = get_predicted_reward(output=new_output, true=new_true)
    new_qval = (1 - alpha) * qval + alpha * (last_reward + gamma * new_reward)
    # todo nie musi zwracac agenta
    # todo zrobic z qval, by byl obliczany w queuenp; czy konieczne?
    return agent, new_output, new_true, new_predicted, new_reward, new_qval


# todo rebuild
def rl_not_a_sarsa(agent, output, x_train, y_train, index, qval, stepsize, tries=100, alpha=0.5, gamma=0.999):
    # todo dwukrotne obliczanie tego samego get_predicted_reward(): tu i na koncu
    _, last_reward = get_predicted_reward(output=output, true=y_train[index])
    # info run tries loops to evaluate the Q value for each of the possible actions
    actions = [-1, +1]
    q = np.zeros(len(actions))
    agent_list = [agent, copy.deepcopy(agent)]
    results = []
    new_output = new_true = new_predicted = new_reward = None
    for k, r in enumerate(actions):
        agent_list[k].reward_step(reward=r, stepsize=stepsize)
        this_gamma = 1
        for m in range(tries):
            p = (index + m + 1) % y_train.shape[0]
            new_obs = get_observation(x_train_obs=x_train[p])
            new_true = y_train[p]
            new_output = agent_list[-1].step(observation=new_obs, output_type="rate", epsilon=1)
            new_predicted, new_reward = get_predicted_reward(output=new_output, true=new_true)
            if m == 0:
                results.append([new_output, new_true, new_predicted, new_reward])
            this_gamma *= gamma
            q[k] += new_reward * this_gamma
    q = q / tries
    # info select agent with higher evaluated q
    best = q.index(max(q))
    agent = agent_list[best]
    new_qval = (1 - alpha) * qval + alpha * (last_reward + gamma * q[best])

    # info agent HAS to be passed back, since reference might have changed
    return agent, new_output, new_true, new_predicted, new_reward, new_qval


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--display", help="display interval", default=10, type=int)
    parser.add_argument("--rl", help="reinforcement method", choices=["naive", "sarsa"], default="naive", type=str)
    args = parser.parse_args()
    mnist_rl_run(args=args)
