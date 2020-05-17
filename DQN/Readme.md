## DQN Algorithm

Implementation of the DQN algorithm with double dueling DQN extension in TensorFlow and OpenAI Gym to get a high score in the Atari Breakout Environemnt.

### Parts of the implementation:
- frame_processor(): to process the input for the DQN
- build_q_network(): to build the ML model
- GameWrapper class: for the Gym environment
- ReplayBuffer class: managing the stored experiences and feeding them to the DQN
- Agent: will take care of choosing the action by putting togeter the Keras model and the ReplayBufer

### Explanation of the files:
- train_dqn.py: for training the dqn
- config.py: parameters and cofig for the model and environment
- evaluation.py: load the agent and watch it play
- visualize.py: visualize the DQNs parameters as it plays
- visualize-replaybuffer: plot the DQNs experiences in a scatterplot

### References:
https://github.com/fg91/Deep-Q-Learning/blob/master/DQN.ipynb
https://www.youtube.com/watch?v=5fHngyN8Qhw
