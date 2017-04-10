# Just Another DAgger implementation

DAgger is a reinforcement learning (imitation learning, to be exact) algorithm that uses data aggregation techniques to address the states distribution mismatch problem. The detailed algorithm is described in the [paper](https://arxiv.org/abs/1011.0686).

This is my implementation of the DAgger. The code is based on the starter code and olicy function generously provided by the [Berkeley CS294 course](https://github.com/berkeleydeeprlcourse/homework).

The following figure plots the mean reward by DAgger over the iterations. DAgger is able to achieve expert-level performance after a few data aggregation process.  
![](dagger_humanoid.png)

(More coming soon...)
