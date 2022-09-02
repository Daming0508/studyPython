import numpy as np
import pandas as pd
import time

np.random.seed(2)     #产生伪随机数列

N_STATES = 6 #the length of the 1 dimensional world 最开始的距离离宝藏的距离有多少步
ACTIONS = ['left','right']  #available actions
EPSILON = 0.9 #greedy policy
ALPHA = 0.1   #learning rate
LAMBDA = 0.9  #discount factor
MAX_EPISODES = 13  #maximum episodes     13回合
FRESH_TIME = 0.1   #fresh time for one move   走一步的时间，为了看到效果

def build_q_table(n_states,actions):                        #建立Q表
    table = pd.DataFrame(
        np.zeros((n_states,len(actions))), #q_table initial values
        columns = actions,     #actions' name
    )
    # print(table)
    return table

# build_q_table(N_STATES,ACTIONS)


#在第一次写代码时，选择动作板块出现bug
def choose_action(state, q_table):
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):  # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)
    else:   # act greedy
        action_name = state_actions.idxmax()    # replace argmax to idxmax as argmax means a different function in newer version of pandas
    return action_name


def get_env_feedback(S,A):                                   #环境奖励
    #This is how agent will interact with the environment
    if A == 'right':          #向右移动
        if S ==N_STATES - 2:
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:                     #向左移动
        R = 0
        if S == 0:
            S_ = S    #reach the wall
        else:
            S_ = S - 1
    return S_,R

def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)



def rl():
    #强化学习主循环
    q_table = build_q_table(N_STATES,ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0   #把探索者放在最左边
        is_terminated = False
        update_env(S,episode,step_counter)
        while not is_terminated:

            A = choose_action(S,q_table)
            S_, R = get_env_feedback(S,A)  #take action&get next state and reward
            q_predict = q_table.loc[S, A]
            if S_ != 'terminal':
                q_target = R + LAMBDA * q_table.iloc[S_,:].max()   #下一步不是终点
            else:
                q_target = R                           # 下一步是终点
                is_terminated = True                  #terminate this episode
            q_table.loc[S, A] +=ALPHA * (q_target - q_predict)   # 进行更新
            S = S_ #移动到下个位置

            update_env(S, episode,step_counter+1)
            step_counter +=1
    return  q_table

if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)