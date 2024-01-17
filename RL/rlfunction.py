import gym
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt, animation
from IPython.display import HTML

def create_animation(frames):
    fig = plt.figure(figsize=(5, 5))
    plt.axis('off')
    ims = []

    for i in range(len(frames)):
        im = plt.imshow(frames[i], animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    plt.close()
    return ani

def env_viz(model, state_size, discrete_actions):
    frames = []
    reward_arr = []
    env = gym.make('Pendulum-v1', render_mode='rgb_array')
    
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    state = np.reshape(state, [1, state_size])

    truncated = False
    while not truncated:
        action_index = np.argmax(model.model.predict(state, verbose=0)[0])
        action = [discrete_actions[action_index]]
        next_state, reward, terminated, truncated, *_ = env.step(action)
        reward_arr.append(reward)
        next_state = np.reshape(next_state, [1, state_size])
        frame = env.render()
        frames.append(frame)
        state = next_state

    env.close()
    ani = create_animation(frames)
    print(f"Reward for test episode: {np.mean(reward_arr)}")
    return HTML(ani.to_jshtml())

def plot_results(score_arr, name_arr, given_name):
    try:
        plt.figure(figsize=(20, 10))
        if len(score_arr) > 1:
            for scores, target_name in zip(score_arr, name_arr):
                sns.lineplot(data = scores, label = target_name)
            plt.suptitle(f"Comparing Average Reward per Episode per model", fontsize=15, y = 0.92)
            plt.legend()
        else:
            sns.lineplot(data = score_arr, label = given_name)
            plt.suptitle(f"Average Reward per Episode, using {given_name} model", fontsize=15, y = 0.92)
        plt.xlim(0,300)
        plt.ylim(None, 0)
        plt.show()
    except TypeError:
        print("Please input as a array.")

def test():
    print("hello world")