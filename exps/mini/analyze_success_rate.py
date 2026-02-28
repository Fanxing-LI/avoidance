#!/usr/bin/env python3
"""
Analyze success rate from batch results for mini scene combinations.
Focuses on:
- Velocities: 2, 4, 6, 8
- Densities: 0.16, 0.25 (mini)

Metrics:
- Success Rate: 1 - collision rate
- Completion Rate (Progress): Average progress towards endpoint (max_x / 56), including agents that collided
- Average Velocity (Effective Period): Average x-direction velocity (vx) when vx > 0.1; Max Velocity: max vx over trajectory
- Average Jerk: Average jerk magnitude (derivative of acceleration)
"""

import os
import re
import argparse
import numpy as np
from pathlib import Path

def calculate_success_from_files(stats_dir, density, speed, endpoint_x=55.0):
    """
    Calculate success rate directly from .npz files.
    Success logic: Agent reaches endpoint_x (55.0) BEFORE or AT THE SAME TIMESTEP as any collision.
    """
    total_agents = 0
    success_count = 0
    
    # Store IDs of failed agents
    stuck_agents = []   # No collision, but didn't reach target
    collided_agents = [] # Collided before reaching target
    
    # Check all scenes (0-4)
    for scene_idx in range(1):  # mini has only 1 scene instance
        # Try parallel run first
        parallel_path = os.path.join(stats_dir, f"stats_d{density:.2f}_s{speed:.1f}_scene{scene_idx}_parallel.npz")
        
        agent_paths = []
        if os.path.exists(parallel_path):
            agent_paths = [parallel_path]
            is_parallel = True
        else:
            # Try individual agent runs
            for agent_idx in range(5):
                p = os.path.join(stats_dir, f"stats_d{density:.2f}_s{speed:.1f}_scene{scene_idx}_agent{agent_idx}.npz")
                if os.path.exists(p):
                    agent_paths.append(p)
            is_parallel = False
            
        for path_idx, p in enumerate(agent_paths):
            data = np.load(p)
            states = data['states']  # (time_steps, num_agents, 28)
            is_col = data['is_col']  # (time_steps, num_agents)
            
            # Handle shape dimensions
            if len(states.shape) == 2:
                states = states[:, np.newaxis, :]
            if len(is_col.shape) == 1:
                is_col = is_col[:, np.newaxis]
                
            positions = states[:, :, 0:3]
            num_agents_in_file = positions.shape[1]
            total_agents += num_agents_in_file
            
            for i in range(num_agents_in_file):
                # Determine Agent ID for reporting
                if is_parallel:
                    agent_id_str = f"Scene {scene_idx} Agent {i}"
                else:
                    agent_id_str = f"Scene {scene_idx} Agent {path_idx}" 
                    
                agent_x = positions[:, i, 0]
                agent_col = is_col[:, i]
                
                # timestamps where x >= endpoint
                success_indices = np.where(agent_x >= endpoint_x)[0]
                
                # timestamps where collision occurred
                collision_indices = np.where(agent_col > 0.5)[0]
                
                if len(success_indices) > 0:
                    first_success_idx = success_indices[0]
                    
                    if len(collision_indices) > 0:
                        first_collision_idx = collision_indices[0]
                        
                        if first_success_idx <= first_collision_idx:
                            success_count += 1
                        else:
                             collided_agents.append(agent_id_str)
                    else:
                        # Reached goal, no collision ever
                        success_count += 1
                else:
                     if len(collision_indices) == 0:
                        # Calculate max_x for stuck agent
                        max_x = np.max(agent_x)
                        stuck_agents.append(f"{agent_id_str} ({max_x:.2f}m)")
                     else:
                        collided_agents.append(agent_id_str)

    if total_agents == 0:
        return None

    return {
        'success_rate': success_count / total_agents,
        'total_agents': total_agents,
        'success_count': success_count,
        'failed_agents': total_agents - success_count,
        'stuck_list': stuck_agents,
        'collided_list': collided_agents
    }


def calculate_success_from_is_success(stats_dir, density, speed):
    """
    Calculate success rate using the is_success field saved by batch_run.
    is_success = 1 means success (no collision), 0 means collision (based on once_collided).
    """
    total_agents = 0
    success_count = 0

    stuck_agents = []
    collided_agents = []

    for scene_idx in range(1):  # mini has only 1 scene instance
        parallel_path = os.path.join(stats_dir, f"stats_d{density:.2f}_s{speed:.1f}_scene{scene_idx}_parallel.npz")

        agent_paths = []
        if os.path.exists(parallel_path):
            agent_paths = [parallel_path]
            is_parallel = True
        else:
            for agent_idx in range(5):
                p = os.path.join(stats_dir, f"stats_d{density:.2f}_s{speed:.1f}_scene{scene_idx}_agent{agent_idx}.npz")
                if os.path.exists(p):
                    agent_paths.append(p)
            is_parallel = False

        for path_idx, p in enumerate(agent_paths):
            data = np.load(p)

            # Use is_success field if available (type2 col index)
            if 'is_success' in data:
                is_success = data['is_success']
                states = data['states']
                is_col = data['is_col']

                if len(states.shape) == 2:
                    states = states[:, np.newaxis, :]
                if len(is_col.shape) == 1:
                    is_col = is_col[:, np.newaxis]

                positions = states[:, :, 0:3]
                num_agents = len(is_success)
                total_agents += num_agents
                success_count += int(np.sum(is_success))
                for i in range(num_agents):
                    if is_success[i] == 1:
                        pass  # Success
                    else:
                        # Find collision x-coordinate
                        agent_col = is_col[:, i]
                        agent_x = positions[:, i, 0]
                        collision_indices = np.where(agent_col > 0.5)[0]
                        if len(collision_indices) > 0:
                            collision_x = agent_x[collision_indices[0]]
                            collided_agents.append(f"Scene {scene_idx} Agent {i} (x={collision_x:.2f}m)")
                        else:
                            collided_agents.append(f"Scene {scene_idx} Agent {i}")
            else:
                # Fallback to old logic
                states = data['states']
                is_col = data['is_col']

                if len(states.shape) == 2:
                    states = states[:, np.newaxis, :]
                if len(is_col.shape) == 1:
                    is_col = is_col[:, np.newaxis]

                positions = states[:, :, 0:3]
                num_agents = positions.shape[1]
                total_agents += num_agents

                for i in range(num_agents):
                    agent_x = positions[:, i, 0]
                    agent_col = is_col[:, i]

                    success_indices = np.where(agent_x >= 55.0)[0]
                    collision_indices = np.where(agent_col > 0.5)[0]

                    if len(success_indices) > 0:
                        first_success_idx = success_indices[0]
                        if len(collision_indices) > 0:
                            first_collision_idx = collision_indices[0]
                            if first_success_idx <= first_collision_idx:
                                success_count += 1
                            else:
                                collision_x = agent_x[first_collision_idx]
                                collided_agents.append(f"Scene {scene_idx} Agent {i} (x={collision_x:.2f}m)")
                        else:
                            success_count += 1
                    else:
                        if len(collision_indices) == 0:
                            stuck_agents.append(f"Scene {scene_idx} Agent {i} ({np.max(agent_x):.2f}m)")
                        else:
                            first_collision_idx = collision_indices[0]
                            collision_x = agent_x[first_collision_idx]
                            collided_agents.append(f"Scene {scene_idx} Agent {i} (x={collision_x:.2f}m)")

    if total_agents == 0:
        return None

    return {
        'success_rate': success_count / total_agents,
        'total_agents': total_agents,
        'success_count': success_count,
        'failed_agents': total_agents - success_count,
        'stuck_list': stuck_agents,
        'collided_list': collided_agents
    }

def calculate_completion_rate(stats_dir, density, speed, endpoint_x=56.0):
    """
    Calculate completion rate (progress) from npz stats files.
    Progress: Average of max_x / endpoint_x for all agents, including those that collided.
    For agents that collided, only count progress up to the collision point (not after).
    Even if an agent collides at x=34, progress is 34/56 = 60.7%.
    """
    all_progress = []
    
    # Check all scenes (0-4) and all agents (0-4 or parallel)
    for scene_idx in range(1):  # mini has only 1 scene instance
        # Try parallel run first
        parallel_path = os.path.join(stats_dir, f"stats_d{density:.2f}_s{speed:.1f}_scene{scene_idx}_parallel.npz")
        if os.path.exists(parallel_path):
            data = np.load(parallel_path)
            states = data['states']  # (time_steps, num_agents, 28)
            is_col = data['is_col']  # (time_steps, num_agents)
            
            num_agents = states.shape[1]
            positions = states[:, :, 0:3]  # (time_steps, num_agents, 3)
            
            for agent_idx in range(num_agents):
                agent_positions = positions[:, agent_idx, :]  # (time_steps, 3)
                agent_collisions = is_col[:, agent_idx]  # (time_steps,)
                
                # Find collision timestep (first timestep where collision occurred)
                collision_timesteps = np.where(agent_collisions > 0.5)[0]
                if len(collision_timesteps) > 0:
                    # Agent collided: only count progress up to collision point
                    collision_timestep = collision_timesteps[0]
                    # Use positions up to and including collision timestep
                    valid_positions = agent_positions[:collision_timestep+1, 0]
                else:
                    # Agent didn't collide: use all positions
                    valid_positions = agent_positions[:, 0]
                
                # Find maximum x position reached in valid portion
                max_x = np.max(valid_positions)
                
                # Calculate progress: max_x / endpoint_x, capped at 100%
                progress = min(max_x / endpoint_x, 1.0)
                all_progress.append(progress)
        else:
            # Try individual agent runs
            for agent_idx in range(5):
                agent_path = os.path.join(stats_dir, f"stats_d{density:.2f}_s{speed:.1f}_scene{scene_idx}_agent{agent_idx}.npz")
                if os.path.exists(agent_path):
                    data = np.load(agent_path)
                    states = data['states']  # (time_steps, 1, 28) or (time_steps, num_agents, 28)
                    is_col = data['is_col']
                    
                    # Handle both single agent and multi-agent cases
                    if len(states.shape) == 2:
                        states = states[:, np.newaxis, :]
                    if len(is_col.shape) == 1:
                        is_col = is_col[:, np.newaxis]
                    
                    num_agents = states.shape[1]
                    positions = states[:, :, 0:3]
                    
                    for a_idx in range(num_agents):
                        agent_positions = positions[:, a_idx, :]
                        agent_collisions = is_col[:, a_idx]
                        
                        # Find collision timestep
                        collision_timesteps = np.where(agent_collisions > 0.5)[0]
                        if len(collision_timesteps) > 0:
                            # Agent collided: only count progress up to collision point
                            collision_timestep = collision_timesteps[0]
                            valid_positions = agent_positions[:collision_timestep+1, 0]
                        else:
                            # Agent didn't collide: use all positions
                            valid_positions = agent_positions[:, 0]
                        
                        # Find maximum x position reached in valid portion
                        max_x = np.max(valid_positions)
                        
                        # Calculate progress: max_x / endpoint_x, capped at 100%
                        progress = min(max_x / endpoint_x, 1.0)
                        all_progress.append(progress)
    
    if len(all_progress) == 0:
        return None
    
    # Return average progress, number of agents that reached endpoint (progress >= 1.0), and total agents
    # Note: Individual agent progress is already capped at 100% before averaging
    avg_progress = np.mean(all_progress)
    completed_count = sum(1 for p in all_progress if p >= 1.0)
    total_agents = len(all_progress)

    return avg_progress, completed_count, total_agents

def calculate_avg_velocity_effective(stats_dir, density, speed, velocity_threshold=0.1):
    """
    Calculate average x-direction velocity (vx) during effective (active movement) periods.
    Effective period: when vx > threshold.
    """
    all_velocities = []
    
    for scene_idx in range(1):  # mini has only 1 scene instance
        parallel_path = os.path.join(stats_dir, f"stats_d{density:.2f}_s{speed:.1f}_scene{scene_idx}_parallel.npz")
        if os.path.exists(parallel_path):
            data = np.load(parallel_path)
            states = data['states']
            velocities = states[:, :, 7:10]  # (time_steps, num_agents, 3)
            
            num_agents = velocities.shape[1]
            for agent_idx in range(num_agents):
                agent_vel = velocities[:, agent_idx, :]  # (time_steps, 3)
                vx = agent_vel[:, 0]
                effective_mask = vx > velocity_threshold
                if np.any(effective_mask):
                    all_velocities.extend(vx[effective_mask].tolist())
        else:
            for agent_idx in range(5):
                agent_path = os.path.join(stats_dir, f"stats_d{density:.2f}_s{speed:.1f}_scene{scene_idx}_agent{agent_idx}.npz")
                if os.path.exists(agent_path):
                    data = np.load(agent_path)
                    states = data['states']
                    
                    if len(states.shape) == 2:
                        states = states[:, np.newaxis, :]
                    
                    velocities = states[:, :, 7:10]
                    num_agents = velocities.shape[1]
                    
                    for a_idx in range(num_agents):
                        agent_vel = velocities[:, a_idx, :]
                        vx = agent_vel[:, 0]
                        effective_mask = vx > velocity_threshold
                        if np.any(effective_mask):
                            all_velocities.extend(vx[effective_mask].tolist())
    
    if len(all_velocities) == 0:
        return None
    
    return np.mean(all_velocities)

def calculate_avg_jerk(stats_dir, density, speed, dt=0.03):
    """
    Calculate average jerk magnitude.
    Jerk = derivative of acceleration = (acc[t+1] - acc[t]) / dt
    """
    all_jerks = []
    
    for scene_idx in range(1):  # mini has only 1 scene instance
        parallel_path = os.path.join(stats_dir, f"stats_d{density:.2f}_s{speed:.1f}_scene{scene_idx}_parallel.npz")
        if os.path.exists(parallel_path):
            data = np.load(parallel_path)
            states = data['states']
            accelerations = states[:, :, 13:16]  # (time_steps, num_agents, 3)
            
            num_agents = accelerations.shape[1]
            for agent_idx in range(num_agents):
                agent_acc = accelerations[:, agent_idx, :]  # (time_steps, 3)
                
                # Calculate jerk as derivative of acceleration
                if len(agent_acc) > 1:
                    jerk = np.diff(agent_acc, axis=0) / dt  # (time_steps-1, 3)
                    jerk_magnitude = np.linalg.norm(jerk, axis=1)  # (time_steps-1,)
                    all_jerks.extend(jerk_magnitude.tolist())
        else:
            for agent_idx in range(5):
                agent_path = os.path.join(stats_dir, f"stats_d{density:.2f}_s{speed:.1f}_scene{scene_idx}_agent{agent_idx}.npz")
                if os.path.exists(agent_path):
                    data = np.load(agent_path)
                    states = data['states']
                    
                    if len(states.shape) == 2:
                        states = states[:, np.newaxis, :]
                    
                    accelerations = states[:, :, 13:16]
                    num_agents = accelerations.shape[1]
                    
                    for a_idx in range(num_agents):
                        agent_acc = accelerations[:, a_idx, :]
                        if len(agent_acc) > 1:
                            jerk = np.diff(agent_acc, axis=0) / dt
                            jerk_magnitude = np.linalg.norm(jerk, axis=1)
                            all_jerks.extend(jerk_magnitude.tolist())
    
    if len(all_jerks) == 0:
        return None
    
    return np.mean(all_jerks)

def calculate_max_velocity(stats_dir, density, speed):
    """
    Calculate average of each agent's max x-direction velocity (vx) over the trajectory.
    """
    all_max_velocities = []
    
    for scene_idx in range(1):  # mini has only 1 scene instance
        parallel_path = os.path.join(stats_dir, f"stats_d{density:.2f}_s{speed:.1f}_scene{scene_idx}_parallel.npz")
        if os.path.exists(parallel_path):
            data = np.load(parallel_path)
            states = data['states']
            velocities = states[:, :, 7:10]  # (time_steps, num_agents, 3)
            
            num_agents = velocities.shape[1]
            for agent_idx in range(num_agents):
                agent_vel = velocities[:, agent_idx, :]  # (time_steps, 3)
                vx = agent_vel[:, 0]
                if len(vx) > 0:
                    all_max_velocities.append(np.max(vx))
        else:
            for agent_idx in range(5):
                agent_path = os.path.join(stats_dir, f"stats_d{density:.2f}_s{speed:.1f}_scene{scene_idx}_agent{agent_idx}.npz")
                if os.path.exists(agent_path):
                    data = np.load(agent_path)
                    states = data['states']
                    
                    if len(states.shape) == 2:
                        states = states[:, np.newaxis, :]
                    
                    velocities = states[:, :, 7:10]
                    num_agents = velocities.shape[1]
                    
                    for a_idx in range(num_agents):
                        agent_vel = velocities[:, a_idx, :]
                        vx = agent_vel[:, 0]
                        if len(vx) > 0:
                            all_max_velocities.append(np.max(vx))
    
    if len(all_max_velocities) == 0:
        return None
    
    # Return average of the max velocities
    return np.mean(all_max_velocities)

def analyze_success_rates(results_dir):
    """Analyze success rates and additional metrics for mini combinations (densities 0.16, 0.25)."""
    # Mini target combinations
    densities = [0.16, 0.25]
    speeds = [2, 4, 6, 8]
    
    # Get stats directory (parent directory)
    stats_dir = os.path.join(os.path.dirname(results_dir), "batch_results_stats")
    
    # Algorithm name
    algo_name = "Ours"
    
    all_rows = []
    
    for density in densities:
        for speed in speeds:
            # Format: d0.16_s3.0
            folder_name = f"d{density:.2f}_s{speed:.1f}"
            # Calculate Metrics using is_success field (type2 col index - once_collided)
            data = calculate_success_from_is_success(stats_dir, density, speed)
            
            if data:
                # Calculate additional metrics
                completion_result = calculate_completion_rate(stats_dir, density, speed)
                avg_vel = calculate_avg_velocity_effective(stats_dir, density, speed)
                max_vel = calculate_max_velocity(stats_dir, density, speed)
                avg_jerk = calculate_avg_jerk(stats_dir, density, speed)
                
                success_count = data['success_count']
                total_runs = data['total_agents']
                success_rate = data['success_rate']
                stuck_list = data.get('stuck_list', [])
                collided_list = data.get('collided_list', [])
                
                row = {
                    "Algo": algo_name,
                    "Vel": speed,
                    "Dens": density,
                    "SuccessRate": success_rate,
                    "SuccessCount": success_count,
                    "TotalRuns": total_runs,
                    "Comp": completion_result[0] if completion_result else 0.0,
                    "AvgVel": avg_vel if avg_vel else 0.0,
                    "MaxVel": max_vel if max_vel else 0.0,
                    "Jerk": avg_jerk if avg_jerk else 0.0,
                    "StuckList": stuck_list,
                    "CollidedList": collided_list
                }
                all_rows.append(row)
    
    if not all_rows:
        print("No results found.")
        return []
        
    # Group rows by Density
    unique_densities = sorted(list(set(r["Dens"] for r in all_rows)))
    
    # Define Column Widths
    # Algo | Vel | Success Rate | Comp | AvgVel | MaxVel | Jerk
    header_fmt = "{:<12} | {:<5} | {:<16} | {:<7} | {:<8} | {:<8} | {:<8}"
    row_fmt    = "{:<12} | {:<5} | {:<16} | {:<7} | {:<8} | {:<8} | {:<8}"
    
    output_lines = []

    for dens in unique_densities:
        section_rows = [r for r in all_rows if r["Dens"] == dens]
        # Sort by Velocity (asc)
        section_rows.sort(key=lambda x: x["Vel"])
        
        # Section Header
        output_lines.append("")
        output_lines.append(f"=== Density: {dens} ===")
        
        # Table Header
        header_str = header_fmt.format("Algo", "Vel", "Success Rate", "Comp", "AvgVel", "MaxVel", "Jerk")
        sep_str = "-" * len(header_str)
        
        output_lines.append(sep_str)
        output_lines.append(header_str)
        output_lines.append(sep_str)
        
        for r in section_rows:
            success_str = f"{r['SuccessCount']}/{r['TotalRuns']} ({r['SuccessRate']:.2%})"
            comp_str = f"{r['Comp']:.3f}"
            vel_str = f"{r['AvgVel']:.2f}"
            max_vel_str = f"{r['MaxVel']:.2f}"
            jerk_str = f"{r['Jerk']:.2f}"
            
            line = row_fmt.format(
                r['Algo'], 
                r['Vel'], 
                success_str,
                comp_str, 
                vel_str, 
                max_vel_str,
                jerk_str
            )
            output_lines.append(line)
            
            # Print failed samples if any
            if r['SuccessCount'] < r['TotalRuns']:
                if r['StuckList']:
                    output_lines.append(f"  [Failed] Stuck: {', '.join(r['StuckList'])}")
                if r['CollidedList']:
                    output_lines.append(f"  [Failed] Collided: {', '.join(r['CollidedList'])}")
            
        output_lines.append(sep_str)

    # Print to console
    print("\n".join(output_lines))
    
    # Save to file (mini summary)
    output_file = Path(stats_dir) / "summary_mini_consolidated.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        f.write("\n".join(output_lines))
    print(f"\nSummary table saved to {output_file}")
    
    return all_rows

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze success rates for Mini experiments.")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Results directory path (default: saved/navigation/batch_results)")
    
    args = parser.parse_args()
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Results directory
    if args.results_dir:
        results_dir = args.results_dir
    else:
        results_dir = os.path.join(script_dir, "saved/navigation/batch_results")
    
    analyze_success_rates(results_dir)
