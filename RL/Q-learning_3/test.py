from maze_env import Maze              #环境
from RLBRAIN import QLearningTable    #大脑


def update():
    for episode in range(100):       #一百个回合
        # initial observation
        obeservation = env.reset()        #环境给出的观测值信息

        while True:
            #fresh env
            env.render()  #刷新环境

            #RL choose action based on observation
            action = RL.choose_action(str(obeservation))      #基于观察（state），RL选择一个动作

            #RL take action and get next observation and reward
            obeservation_ , reward, done = env.step(action)   #在环境中施加动作会返回一个新的观测值

            #RL learn from this transition
            RL.learn(str(obeservation),action,reward,str(obeservation_))

            #swap observation
            obeservation = obeservation_

            #break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100,update())
    env.mainloop()
