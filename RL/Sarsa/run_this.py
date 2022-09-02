from maze_env import Maze
from RLBRAIN import SarsaTable

def update():
    for episode in range(100):
        # initial observation
        observation = env.reset()

        # RL choose action based on observation            与Q-learning不同，Sarsa选择动作是在这里选的
        action = RL.choose_action(str(observation))

        while True:
            #fresh env
            env.render()

            #RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            #RL choose action based on next observation  与Q-learning不同，Q只会估计下一个action并不一定会采取，但是Sarsa一定会采取
            action_ = RL.choose_action(str(observation_))

            # RL learn from this transition (s, a, r, s, a) →Sarsa  与Qlearning不同之处，需要考虑下一个action
            RL.learn(str(observation), action, reward, str(observation_),action_)

            # swap observation and action
            observation = observation_
            action = action_

            #break while loop when end of this episode
            if done:
                break

        # end of game
        print('game over')
        env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = SarsaTable(actions=list(range(env.n_actions)))

    env.after(100, update())
    env.mainloop()
