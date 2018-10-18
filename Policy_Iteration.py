# package
import numpy as np
import time, os

## define function
def PolicyEvalution(func_value, best_action, func_reward, trans_mat, gamma):
    func_value_now = func_value.copy()
    for state in range(1,15):
        next_state = trans_mat[:, state, best_action[state]]
        future_reward = func_reward + func_value_now*gamma
        func_value[state] = np.sum(next_state*future_reward)
    delta = np.max(np.abs(func_value - func_value_now))
    return func_value, delta

def ShowValue(delta, theta, gamma, counter, func_value):
    os.system('cls' if os.name == 'nt' else 'clear')
    print('='*60)
    print('[Parameters]')
    print('Gamma = ' + str(gamma))
    print('Threshold = ' + str(theta) + '\n')
    print('[Variables]')
    print('No.' + str(counter) + ' iteration')
    print('Delta = ' +str(delta) + '\n')
    print('[State-Value]')
    print(func_value.reshape(4,4))
    print('='*60)

def PolicyImprovement(func_value, best_action, prob_action, func_reward, trans_mat, gamma):
    policy_stable = False
    best_action_now = best_action.copy()
    for state in range(1,15):
        prob_next_state = prob_action[state]*trans_mat[:,state,:]
        future_reward = func_reward + func_value*gamma
        best_action[state] = np.argmax(np.matmul(np.transpose(prob_next_state), future_reward))
    if np.all(best_action == best_action_now):
        policy_stable = True
    return best_action, policy_stable

# main function
def main():
    ## environment setting
    # action
    BestAction = np.random.randint(0,4,16)
    ProbAction = np.zeros([16,4])
    ProbAction[1:15,:] = 0.25
    # value function
    FuncValue = np.zeros(16)
    # reward function
    FuncReward = np.full(16,-1)
    FuncReward[0] = 0
    FuncReward[15] = 0
    # transition matrix
    T = np.load('./gridworld/T.npy')

    # parameters
    delta = 0.1
    gamma = 0.99
    theta = 0.05
    counter = 1

    # iteration
    while delta > theta:
        FuncValue, delta = PolicyEvalution(FuncValue, BestAction, FuncReward, T, gamma)
        ShowValue(delta, theta, gamma, counter, FuncValue)
        counter += 1
        time.sleep(1)

    BestAction, PolicyStable = PolicyImprovement(FuncValue, BestAction, ProbAction, FuncReward, T, gamma)


# execute
if __name__ == '__main__':
    main()


