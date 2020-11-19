# RC2020

[ML Reproducibility Challenge 2020](https://paperswithcode.com/rc2020) is a community challenge for machine learning enthusiasts, students, and researchers in which participants select a paper from one of the prestigious ML conferences of the year (NeurIPS, ICML, ICLR, ACL, EMNLP, CVPR or ECCV). In an attempt to relicate the paper, participants provide additional support either towards or against the claims and results of the work. Alongside the goal of evaluating the validity/legitemacy of recent research, this project primary serves to assess the reproducibility/replicability of Machine Learning research.

Here is a collection of my work-in-progress code for the challenge. I am working on [*Discovering Reinforcement Learning Algorithms*](https://arxiv.org/pdf/2007.08794v1.pdf) by [DeepMind](https://deepmind.com/research/publications/Discovering-Reinforcement-Learning-Algorithms).

This code is not expected to be very well organized until towards the end of the project. I am merely dropping the source code thus far into this repo each time I update it.

## Progress Log
**Thurday, November 5, 2020**:
* Implemented all Grid World environments, including Tabular Grid World and Random Grid World, and the five maps for each type of grid world.
<img src="https://i.ibb.co/QMmJzZ5/Screen-Shot-2020-11-05-at-6-08-24-PM.png" alt="GridWorld" border="0">

**Sunday, November 8, 2020**
* Implemented all Delayed Chain MDP environments, which includes 4 standard maps and 1 unique mode.
<img src="https://i.ibb.co/j3wxSKV/Screen-Shot-2020-11-08-at-6-34-24-PM.png" alt="DelayedChainMDP" border="0">

**Saturday, November 14, 2020**
* Rewrote the Grid World environments, doubling the speed of simulation and improving the rendering graphics
<img src="https://i.ibb.co/2YnGksh/Screen-Shot-2020-11-14-at-4-20-44-PM.png" alt="GridWorld" border="0">

**Tuesday, November 17, 2020**
* Wrote the `Agent` abstract class, and all subsequent agent concrete classes, including `TabularAgent` (for the Tabula Grid World environment), `FunctionalAgent` (for environments demanding function approximation, ie. Random Grid World and Delayed Chain MDP + State Distraction), and `BinaryAgent` (for the standard Delayed Chain MDP environments without state distraction).

**Wednesday, November 18, 2020**
* Wrote the `LPG` Model class and the Embedding layer it uses to encode the categorical prediction vector <img src="https://render.githubusercontent.com/render/math?math=y">

## Task Log
- [x] Read DeepMind's [*Discovering Reinforcement Learning Algorithms*](https://arxiv.org/pdf/2007.08794v1.pdf)
- [x] Write the gym Class for the custom Grid World environments
- [x] Write the gym Class for the custom MDP environments
- [ ] Fix a rendering bug in the Random Grid World environment that causes some squares to remain lit although the reward located on that square was already collected.
- [x] Write a custom TensorFlow model for the Learned Policy Gradient architecture
- [x] Write some classes for the various agent structures for each training environment

  * [x] `Agent` abstract class
  * [x] `TabularAgent` for Tabular Grid World
  * [x] `FunctionalAgent` for Random Grid World and Delayed Chain MDP + State Distraction
  * [x] `BinaryAgent` for Delayed Chain MDP
  
- [ ] Implement the agent update
- [ ] Implement the Learned Policy Gradient algorithm
- [ ] Train on each environment
- [ ] Test the learned update rule on each Atari environment
