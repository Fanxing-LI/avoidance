

# Simple but Stable, Fast and Safe: Achieve End-to-end Control by High-Fidelity Differentiable Simulation

Obstacle avoidance is a fundamental vision-based task essential for enabling quadrotors to perform advanced applications. When planning the trajectory, existing approaches both on optimization and learning typically regard quadrotor as a point-mass model, giving path or velocity commands then tracking the commands by outer-loop controller.

However, at high speeds, planned trajectories sometimes become dynamically infeasible in actual flight, which beyond the capacity of controller.
Although direct taking low-level bodyrate commands as output can mitigate this issue, it gets much challenging to design such a low-level policy because the transition process is so complex and less smooth that the difficulty of training significantly increases.

In this paper, we propose a novel end-to-end policy that directly maps depth images to low-level bodyrate commands by reinforcement learning via differentiable simulation.
The high-fidelity simulation in training after parameter identification significantly reduces all the gaps between training, simulation and real world.
Analytical process by differentiable simulation provides accurate gradient to ensure efficiently training the low-level policy without expert guidance. 
The policy employs a lightweight and the most simple inference pipeline that runs without explicit mapping, backbone networks, primitives, recurrent structures, or backend controllers, nor curriculum or privileged guidance. By inferring low-level command directly to the hardware controller, the method enables full flight envelope control and avoids the dynamic-infeasible issue.

Experimental results demonstrate that the proposed approach achieves the highest success rate and the lowest jerk among state-of-the-art baselines across multiple benchmarks. The policy also exhibits strong generalization, successfully deploying zero-shot in unseen, outdoor environments while reaching speeds of up to 7.5m/s as well as stably flying in the super-dense forest.
Furthermore, we provide the first successful demonstration of backpropagating image information through high-fidelity differentiable simulation, validating the extensibility of first-order gradient methods to other complex robotic systems.

## Clone repository
clone the avoidance repository and all its submodules:
```bash
git clone --recursive https://github.com/Fanxing-LI/avoidance
```

## Install VisFly
```bash
cd avoidance/VisFly
```
Follow the instructions in [VisFly](https://github.com/SJTU-ViSYS-team/VisFly) to install the VisFly simulator.

## Clone the datasets
```bash
cd avoidance/VisFly/datasets
git clone -b avoidance https://huggingface.co/datasets/LiFanxing/visfly-beta
```

## Train the policy (Or skip this step and download pre-train Model)
```bash
cd avoidance
conda activate visfly
python exps/real_world/run.py -e navigation -a SHAC -t 1
# python exps/real_world/run.py -t 1 -a PPO -e navigation -c comment
python exps/real_world/run.py -e navigation -a SHAC -t 0 -w SHAC_std_1.zip
```
If you define a new comment, replace `std` with your comment in the last command.

## Download the pre-trained weights
[Download](https://drive.google.com/file/d/1vKx2L6aIMCFVegsi_5F9uyyCsQg7XDZm/view?usp=sharing) the pre-trained weights and put it in `avoidance/exps/real_world/saved/navigation/`.
```bash
cd avoidance
conda activate visfly
python exps/real_world/run.py -e navigation -a SHAC -t 0 -w checkpoint.zip
```

## Check the test result
The test result will be saved at `exps/real_world/saved/navigation/test/checkpoint/`. You can adjust the content of saved figure and video in `exps/test.py` and rendering configuration at `eval_env.scene_kwargs.render_settings` in `exps/real_world/env_cfgs/navigation.yaml`.

## Finetune the policy
If you wanna to finetune the policy, create another env or directly modify NavigationEnv.get_reward(). Then
```bash
python exps/real_world/run.py  -e navigation  -a SHAC -t 1 -w checkpoint.zip
```
Or you wanna train the checkpoint using another algorithm:
```bash
python exps/real_world/run.py  -e navigation  -a BPTT -t 1 -w checkpoint.zip
```
