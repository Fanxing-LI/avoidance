import os
import sys
import numpy as np
import torch as th
import argparse
import subprocess
import matplotlib
matplotlib.use("Agg")  # non-interactive backend so savefig works in subprocess/headless
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set reproducibility
SEED = 42
th.manual_seed(SEED)
np.random.seed(SEED)
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = False

# Add project root to path
sys.path.append(os.getcwd())

from VisFly.utils.common import load_yaml_config
from envs.RealNavigationEnv import RealNavigationEnv
from algorithms.BPTT_series.BPTT import BPTT
from exps.test import Test


def run_single_experiment(
    density, speed, parallel=False, deterministic=True, sample=False, scale=1.0, seed=42, comment=""
):
    print(f"Processing: Density={density}, Speed={speed}, Parallel={parallel}")

    # Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    env_cfg_path = os.path.join(current_dir, "env_cfgs/navigation.yaml")
    alg_cfg_path = os.path.join(current_dir, "alg_cfgs/navigation/BPTT.yaml")

    # Weights
    # Available in exps/mini/saved/navigation/
    weights_name = "SHAC_check_checkpoint_H150r_refineRand_refineCol_remap0.2_0.3Share_treVel2_5"
    save_folder = os.path.join(current_dir, "saved/navigation")
    weights_path = os.path.join(save_folder, weights_name)

    # Append comment to output dirs if provided
    batch_results_name = f"batch_results_{comment}" if comment else "batch_results"
    batch_results_stats_name = f"batch_results_stats_{comment}" if comment else "batch_results_stats"

    output_dir = os.path.join(save_folder, batch_results_stats_name)
    os.makedirs(output_dir, exist_ok=True)

    # Load Configs
    env_config = load_yaml_config(env_cfg_path)
    alg_config = load_yaml_config(alg_cfg_path)

    # Modify Env Config for Batch Run (mini scenes: box30_wall_mini{density}tree)
    scene_path = f"VisFly/datasets/visfly-beta/configs/scenes/box30_wall_mini{density}tree"
    env_config["eval_env"]["scene_kwargs"]["path"] = scene_path
    env_config["eval_env"]["max_rand_velocity"] = float(speed)

    # Per-agent targets: pure x-direction velocity [vx, 0, 0]
    target_list = [
        [float(speed), 0.0, 0.0],
        [float(speed), 0.0, 0.0],
        [float(speed), 0.0, 0.0],
        [float(speed), 0.0, 0.0],
        [float(speed), 0.0, 0.0],
    ]

    # Configure 1 scene
    env_config["eval_env"]["num_scene"] = 1

    # Set seed for reproducibility
    env_config["eval_env"]["random_kwargs"]["state_generator"]["kwargs"][0]["seed"] = seed

    if parallel:
        env_config["eval_env"]["num_agent_per_scene"] = 5
        # Use all targets (velocity-based)
        env_config["eval_env"]["target"] = target_list
        runs_per_scene = 1
    else:
        env_config["eval_env"]["num_agent_per_scene"] = 1
        # Use single target for single agent setup
        env_config["eval_env"]["target"] = [target_list[0]]
        runs_per_scene = 5

    # Initialize Environment
    env = RealNavigationEnv(**env_config["eval_env"])

    # Load Model
    model = BPTT.load(weights_path, env=env)

    # Base save path for this experiment setting
    base_exp_path = os.path.join(save_folder, batch_results_name, f"d{density}_s{speed}")

    # Metrics accumulator for aggregation
    agg_failed_agents = 0
    agg_total_agents = 0
    agg_scene_stats = []

    # Mini scenes have only 1 instance; run scene_0 only
    for scene_idx in range(1):
        print(f"  > Running Scene {scene_idx} (d{density}, s{speed})")

        # Load the specific scene
        # __init__ loads the first batch (Scene 0).
        # For subsequent iterations, we explicitly load the next scene.
        if scene_idx > 0:
            env.envs.sceneManager.load_scenes()

        # Create subfolder for this scene
        scene_save_path = os.path.join(base_exp_path, f"scene_{scene_idx}")

        scene_failed = 0
        scene_total = 0

        for agent_idx in range(runs_per_scene):
            if parallel:
                agent_save_path = os.path.join(scene_save_path, "parallel_run")
                # Set all velocity targets
                env.target = th.tensor(
                    target_list, dtype=th.float32, device=env.device
                )
                # For parallel mode (5 agents), let env naturally generate different positions
                # based on xyz_num: [1,5,1] - each agent gets a different starting position (0-4)
                # DON'T reset current_generate_index - let it auto-increment during env reset
                g0_index = None
            else:
                agent_save_path = os.path.join(scene_save_path, f"agent_{agent_idx}")
                # Set single velocity target
                env.target = th.tensor(
                    [target_list[agent_idx]], dtype=th.float32, device=env.device
                )
                # Generator index for single agent mode
                g0_index = int(agent_idx)

            os.makedirs(agent_save_path, exist_ok=True)

            # Ensure each run has its own deterministic birth position (agent0..agent4),
            # matching the previous "5 agents in one scene" setup.
            # We force the UniformStateRandomizer test-grid index before reset().
            assert (
                hasattr(env, "envs")
                and hasattr(env.envs, "stateGenerators")
                and len(env.envs.stateGenerators) > 0
            ), (
                "Expected env.envs.stateGenerators to exist for deterministic test birth positions."
            )
            g0 = env.envs.stateGenerators[0]
            assert getattr(g0, "test", False) and hasattr(
                g0, "current_generate_index"
            ), (
                "Expected eval_env.random_kwargs.state_generator.kwargs to set test=True and xyz_num for deterministic births."
            )
            # Only reset index for single-agent mode; parallel mode uses natural index 0-4
            if g0_index is not None:
                g0.current_generate_index = g0_index

            test_handle = Test(model=model, save_path=agent_save_path, name="")

            # Run test
            test_args = dict(alg_config.get("test", {}))
            with th.no_grad():
                test_handle.test(**test_args)

            # Get collision indices (type2 - based on once_collided)
            # test_handle.is_success = 1 means no collision, 0 means collision
            is_success = test_handle.is_success.cpu().numpy()
            type2_col_indices = np.where(is_success == 0)[0]

            # Post-processing and Saving
            # Collect Data
            states = (
                th.stack(test_handle.state_all).cpu().numpy()
                if test_handle.state_all
                else np.array([])
            )
            rewards = np.array(test_handle.reward_all)

            # Collision Data
            if test_handle.collision_all:
                col_dis = (
                    th.stack([c["col_dis"] for c in test_handle.collision_all])
                    .cpu()
                    .numpy()
                )
                is_col = (
                    th.stack([c["is_col"] for c in test_handle.collision_all])
                    .cpu()
                    .numpy()
                )
                col_pts = (
                    th.stack([c["col_pt"] for c in test_handle.collision_all])
                    .cpu()
                    .numpy()
                )
            else:
                col_dis = np.array([])
                is_col = np.array([])
                col_pts = np.array([])

            # Calculate success timesteps: first timestep where x >= 56 (endpoint) for each agent
            # Success timestep is only valid if agent didn't collide
            success_timesteps = []
            if states.size > 0:
                positions = states[:, :, 0:3]  # (time_steps, num_agents, 3)
                num_agents = positions.shape[1]
                endpoint_x = 56.0

                for agent_idx in range(num_agents):
                    agent_x = positions[:, agent_idx, 0]  # (time_steps,)
                    agent_collisions = (
                        is_col[:, agent_idx]
                        if is_col.size > 0
                        else np.zeros(len(agent_x), dtype=bool)
                    )
                    had_collision = np.any(agent_collisions > 0.5)

                    # Find first timestep where x >= endpoint
                    success_indices = np.where(agent_x >= endpoint_x)[0]
                    if len(success_indices) > 0 and not had_collision:
                        # Success: reached endpoint without collision
                        success_timesteps.append(int(success_indices[0]))
                    else:
                        # No success: either collided or never reached endpoint
                        success_timesteps.append(-1)  # -1 indicates no success
            else:
                success_timesteps = []

            # Save NPZ Stats
            if parallel:
                stats_path = os.path.join(
                    output_dir,
                    f"stats_d{density}_s{speed}_scene{scene_idx}_parallel.npz",
                )
                saved_target = np.array(target_list, dtype=np.float32)
            else:
                stats_path = os.path.join(
                    output_dir,
                    f"stats_d{density}_s{speed}_scene{scene_idx}_agent{agent_idx}.npz",
                )
                saved_target = np.array(
                    target_list[agent_idx], dtype=np.float32
                )

            np.savez_compressed(
                stats_path,
                states=states,
                rewards=rewards,
                col_dis=col_dis,
                is_col=is_col,
                col_pts=col_pts,
                episode_rewards=test_handle.eq_r,
                episode_lengths=test_handle.eq_l,
                target=saved_target,
                success_timesteps=np.array(success_timesteps, dtype=np.int32)
                if success_timesteps
                else np.array([], dtype=np.int32),
                is_success=is_success,
                type2_col_indices=type2_col_indices,
            )

            # Print collision info (matching test.py output)
            print(f"    type2 col index: {type2_col_indices}")

            # --- Analysis & Plotting ---
            if states.size > 0:
                num_agents = states.shape[1]

                # Calculate Failed Agents for this agent-run (type2 - based on once_collided)
                failed_agents_count = len(type2_col_indices)

                # Update Aggregates
                agg_failed_agents += failed_agents_count
                agg_total_agents += num_agents

                scene_failed += failed_agents_count
                scene_total += num_agents

                # Plot
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection="3d")

                for i in range(num_agents):
                    ax.plot(
                        states[:, i, 0],
                        states[:, i, 1],
                        states[:, i, 2],
                        alpha=0.5,
                        linewidth=1.2,
                    )

                col_indices = np.where(is_col > 0.5)
                if len(col_indices[0]) > 0:
                    c_pts = col_pts[col_indices]
                    ax.scatter(
                        c_pts[:, 0],
                        c_pts[:, 1],
                        c_pts[:, 2],
                        c="red",
                        marker="x",
                        s=20,
                        label="Collision",
                    )

                ax.set_xlabel("X (m)")
                ax.set_ylabel("Y (m)")
                ax.set_zlabel("Z (m)")
                ax.set_title(
                    f"Scene {scene_idx} Agent {agent_idx} - Failed: {failed_agents_count}/{num_agents}"
                )

                plt.savefig(os.path.join(agent_save_path, "trajectories_3d.png"))
                plt.close(fig)

                # Text Report for agent-run
                with open(
                    os.path.join(agent_save_path, "collision_count.txt"), "w"
                ) as f:
                    f.write(f"Scene: {scene_idx}\n")
                    if parallel:
                        f.write(f"Mode: Parallel (5 agents)\n")
                        f.write(f"Targets: {target_list}\n")
                    else:
                        f.write(f"Agent: {agent_idx}\n")
                        f.write(f"Target: {target_list[agent_idx]}\n")
                    f.write(f"Failed Agents (type2 - once_collided): {failed_agents_count}\n")
                    f.write(f"type2 col index: {type2_col_indices}\n")
                    f.write(f"Per-Agent (1=Fail, type2 - once_collided):\n")
                    for i in range(num_agents):
                        c_count = 1 if i in type2_col_indices else 0
                        f.write(f"Agent {i}: {c_count}\n")

            # Cleanup Handle (keep env and model for next loop)
            del test_handle

        agg_scene_stats.append(
            {
                "scene_idx": scene_idx,
                "failed": int(scene_failed),
                "total": int(scene_total),
            }
        )

    # --- Aggregate Report for Density/Speed Pair ---
    summary_path = os.path.join(base_exp_path, "summary_collision_report.txt")
    with open(summary_path, "w") as f:
        f.write(f"Aggregate Report: Density={density}, Speed={speed}\n")
        f.write(f"==========================================\n")
        f.write(f"Total Scenes Processed: {len(agg_scene_stats)}\n")
        f.write(f"Total Agents Flown: {agg_total_agents}\n")
        f.write(f"Total Failed Agents: {agg_failed_agents}\n")
        if agg_total_agents > 0:
            f.write(
                f"Overall Failure Rate: {agg_failed_agents / agg_total_agents:.2%}\n"
            )
        else:
            f.write("Overall Failure Rate: N/A\n")
        f.write("\nBreakdown by Scene:\n")
        for stat in agg_scene_stats:
            f.write(
                f"  Scene {stat['scene_idx']}: {stat['failed']}/{stat['total']} Failed\n"
            )
    print(f"Saved aggregate report to {summary_path}")

    # Cleanup Env
    env.close()
    del env
    del model


def run_batch_loop(parallel=False, deterministic=True, sample=False, scale=1.0, seed=42, comment=""):
    """Run batch experiments and analyze results at the end. Mini: densities 0.16, 0.25 only."""
    densities = ["0.16", "0.25"]
    speeds = [2, 4, 6, 8]

    current_script = os.path.abspath(__file__)

    for density in densities:
        for speed in speeds:
            print(
                f"Launching subprocess: Density={density}, Speed={speed}, Parallel={parallel}, Scale={scale}, Seed={seed}, Comment={comment}"
            )
            cmd = [
                sys.executable,
                current_script,
                "--density",
                str(density),
                "--speed",
                str(speed),
                "--scale",
                str(scale),
                "--seed",
                str(seed),
            ]
            if comment:
                cmd.extend(["--comment", comment])
            if parallel:
                cmd.append("--parallel")
            if deterministic:
                cmd.append("--deterministic")
            if sample:
                cmd.append("--sample")
            ret = subprocess.run(cmd, check=False)
            if ret.returncode != 0:
                print(
                    f"Subprocess finished with exit code {ret.returncode} (might be expected GL crash)."
                )

    # After all experiments complete, run analysis
    print("\n" + "=" * 80)
    print("All batch experiments completed. Running analysis...")
    print("=" * 80)

    # Import and run analysis
    script_dir = os.path.dirname(os.path.abspath(__file__))
    analyze_script = os.path.join(script_dir, "analyze_success_rate.py")
    if os.path.exists(analyze_script):
        analyze_cmd = [sys.executable, analyze_script]
        ret = subprocess.run(analyze_cmd, check=False, cwd=script_dir)
        if ret.returncode != 0:
            print(f"Analysis script finished with exit code {ret.returncode}")
    else:
        print(f"Warning: Analysis script not found at {analyze_script}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--density", type=str)
    parser.add_argument("--speed", type=float)
    parser.add_argument(
        "--parallel",
        default=True,
        action="store_true",
        help="Run 5 agents in parallel per scene",
    )
    parser.add_argument(
        "--deterministic",
        default=True,
        action="store_true",
        help="Use deterministic testing (default: True)",
    )
    parser.add_argument(
        "--sample",
        default=False,
        action="store_true",
        help="Sample actions during testing (default: False)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale factor for obstacles (default: 1.0)",
    )
    parser.add_argument(
        "--comment", "-c",
        type=str,
        default="",
        help="Comment to append to output folder names (default: empty)",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    # Always True for batch runs (flags are passed in subprocess calls to avoid argparse errors)
    deterministic = True
    sample = False

    if args.density and args.speed:
        run_single_experiment(
            args.density,
            args.speed,
            parallel=args.parallel,
            deterministic=deterministic,
            sample=sample,
            scale=args.scale,
            seed=args.seed,
            comment=args.comment,
        )
    else:
        run_batch_loop(
            parallel=args.parallel, deterministic=deterministic, sample=sample, scale=args.scale, seed=args.seed, comment=args.comment
        )
