import torch as th
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def transform_distance_by_angle(d, a, min_d, max_d, max_angle_increase=0.8):
    """
    根据距离变换角度，距离越近角度增大越多，使用非线性变换确保角度不超过90度。

    Args:
        d: 距离 (tensor or scalar)
        a: 角度（弧度制） (tensor or scalar)
        min_d: 最小距离阈值
        max_d: 最大距离阈值
        max_angle_increase: 最大角度增加量，以90度的比例表示（0到1之间），默认0.8表示最多增加到接近90度

    Returns:
        transformed angle: 变换后的角度（弧度制）
    """
    d = th.as_tensor(d, dtype=th.float32)
    a = th.as_tensor(a, dtype=th.float32)

    pi_2 = th.pi / 2  # 90度

    # 条件1: 角度 >= 90度 -> 不做变换
    angle_mask = a >= pi_2

    # 条件2: 距离 >= max_d -> 不做变换
    distance_mask = d >= max_d

    # 不变换的情况
    no_transform_mask = angle_mask | distance_mask

    # 计算距离因子：距离越近，因子越大
    # 当 d < min_d 时，使用最大变换
    # 当 min_d <= d < max_d 时，使用渐变变换
    d_clamped = d.clamp(min=min_d, max=max_d)
    d_normalized = (d_clamped - min_d) / (max_d - min_d)  # 0到1之间
    # 使用平方或其他非线性函数，使得距离近时变化更明显
    d_factor = (1.0 - d_normalized) ** 2  # 1.0 at min_d, 0.0 at max_d

    # 对于 d < min_d 的情况，使用最大因子
    d_factor = th.where(d < min_d, th.ones_like(d_factor), d_factor)

    # 计算角度增量
    # 使用非线性映射确保增加后的角度不超过90度
    # 策略：计算从当前角度到90度的"剩余空间"，然后根据距离因子填充这个空间
    angle_gap = pi_2 - a  # 到90度的距离

    # 根据距离因子和max_angle_increase参数决定增加多少
    # 使用指数函数确保平滑过渡，max_angle_increase控制最大增加比例
    angle_increase = angle_gap * max_angle_increase * (1.0 - th.exp(-3.0 * d_factor))

    # 计算变换后的角度
    a_transformed = th.where(
        no_transform_mask,
        a,  # 不变换
        (a + angle_increase).clamp(max=pi_2)  # 增加角度，但只对<90度的角度进行clamp
    )

    return a_transformed


def test_basic_cases():
    """测试基本边界情况"""
    print("=" * 60)
    print("测试基本情况")
    print("=" * 60)

    min_d = 0.2
    max_d = 4.0

    # 测试1: 角度 >= 90度 -> 不做变换
    d = th.tensor([2.0, 3.0, 4.0])
    a = th.tensor([np.pi/2, np.pi/2 + 0.1, np.pi])  # 90°, >90°, 180°
    result = transform_distance_by_angle(d, a, min_d, max_d)
    print(f"\n测试1: 角度 >= 90°（应该不变）")
    print(f"输入距离: {d.numpy()}")
    print(f"输入角度(度): {(a * 180 / np.pi).numpy()}")
    print(f"输出角度(度): {(result * 180 / np.pi).numpy()}")
    print(f"角度变化(度): {((result - a) * 180 / np.pi).numpy()} (应该约为0)")

    # 测试2: 距离 >= max_d -> 不做变换
    d = th.tensor([5.0, 6.0, 10.0])
    a = th.tensor([0.0, np.pi/4, np.pi/6])  # 0°, 45°, 30°
    result = transform_distance_by_angle(d, a, min_d, max_d)
    print(f"\n测试2: 距离 >= max_d（应该不变）")
    print(f"输入距离: {d.numpy()}")
    print(f"输入角度(度): {(a * 180 / np.pi).numpy()}")
    print(f"输出角度(度): {(result * 180 / np.pi).numpy()}")
    print(f"角度变化(度): {((result - a) * 180 / np.pi).numpy()} (应该约为0)")

    # 测试3: 距离 < min_d -> 最大变换
    d = th.tensor([0.5, 0.7, 0.9])
    a = th.tensor([0.0, np.pi/6, np.pi/3])  # 0°, 30°, 60°
    result = transform_distance_by_angle(d, a, min_d, max_d)
    print(f"\n测试3: 距离 < min_d（最大角度增加）")
    print(f"输入距离: {d.numpy()}")
    print(f"输入角度(度): {(a * 180 / np.pi).numpy()}")
    print(f"输出角度(度): {(result * 180 / np.pi).numpy()}")
    print(f"角度增加(度): {((result - a) * 180 / np.pi).numpy()}")
    print(f"验证不超过90度: {(result <= np.pi/2).all().item()}")

    # 测试4: 正常范围，小角度 -> 大幅增加
    d = th.tensor([1.5, 2.5, 4.0])
    a = th.tensor([0.0, np.pi/6, np.pi/4])  # 0°, 30°, 45°
    result = transform_distance_by_angle(d, a, min_d, max_d)
    print(f"\n测试4: 正常距离范围，不同角度")
    print(f"输入距离: {d.numpy()}")
    print(f"输入角度(度): {(a * 180 / np.pi).numpy()}")
    print(f"输出角度(度): {(result * 180 / np.pi).numpy()}")
    print(f"角度增加(度): {((result - a) * 180 / np.pi).numpy()}")
    print(f"验证不超过90度: {(result <= np.pi/2).all().item()}")

    # 测试5: 接近90度的角度
    d = th.tensor([1.0, 2.0, 3.0])
    a = th.tensor([80, 85, 89]) * np.pi / 180  # 80°, 85°, 89°
    result = transform_distance_by_angle(d, a, min_d, max_d)
    print(f"\n测试5: 接近90度的角度（验证不会超过90度）")
    print(f"输入距离: {d.numpy()}")
    print(f"输入角度(度): {(a * 180 / np.pi).numpy()}")
    print(f"输出角度(度): {(result * 180 / np.pi).numpy()}")
    print(f"角度增加(度): {((result - a) * 180 / np.pi).numpy()}")
    print(f"验证不超过90度: {(result <= np.pi/2).all().item()}")

    # 测试6: 90度到180度之间的角度（应该不变）
    d = th.tensor([0.5, 1.5, 2.5, 3.5])
    a = th.tensor([95, 120, 150, 175]) * np.pi / 180  # 95°, 120°, 150°, 175°
    result = transform_distance_by_angle(d, a, min_d, max_d)
    print(f"\n测试6: 90度到180度之间的角度（应该不变）")
    print(f"输入距离: {d.numpy()}")
    print(f"输入角度(度): {(a * 180 / np.pi).numpy()}")
    print(f"输出角度(度): {(result * 180 / np.pi).numpy()}")
    print(f"角度变化(度): {((result - a) * 180 / np.pi).numpy()} (应该约为0)")
    print(f"验证角度不变: {th.allclose(result, a, atol=1e-6)}")


def visualize_2d_slices():
    """可视化不同输入角度下，角度变换如何随距离变化"""
    print("\n" + "=" * 60)
    print("创建2D可视化")
    print("=" * 60)

    min_d = 1.0
    max_d = 5.0

    # 创建距离范围
    distances = th.linspace(0.2, 8.0, 200)

    # 测试不同的输入角度
    angles_deg = [0, 15, 30, 45, 60, 75, 85, 95, 120, 150, 175]
    angles_rad = [np.deg2rad(a) for a in angles_deg]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 图1: 变换后的角度 vs 输入距离
    for angle_deg, angle_rad in zip(angles_deg, angles_rad):
        angles = th.full_like(distances, angle_rad)
        transformed = transform_distance_by_angle(distances, angles, min_d, max_d)
        transformed_deg = transformed * 180 / np.pi
        ax1.plot(distances.numpy(), transformed_deg.numpy(),
                label=f'Input angle={angle_deg}°', linewidth=2)

    # 添加参考线
    ax1.axhline(90, color='k', linestyle='--',
            alpha=0.5, linewidth=2, label='90° limit')
    ax1.axvline(min_d, color='r', linestyle='--', alpha=0.3, linewidth=1)
    ax1.axvline(max_d, color='r', linestyle='--', alpha=0.3, linewidth=1)
    ax1.text(min_d, ax1.get_ylim()[1] * 0.95, f'min_d={min_d}',
            ha='center', fontsize=9, color='r')
    ax1.text(max_d, ax1.get_ylim()[1] * 0.95, f'max_d={max_d}',
            ha='center', fontsize=9, color='r')

    ax1.set_xlabel('Input Distance', fontsize=12)
    ax1.set_ylabel('Transformed Angle (deg)', fontsize=12)
    ax1.set_title('Angle Transformation for Different Input Angles', fontsize=13)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 图2: 角度增加量 vs 输入距离
    for angle_deg, angle_rad in zip(angles_deg, angles_rad):
        angles = th.full_like(distances, angle_rad)
        transformed = transform_distance_by_angle(distances, angles, min_d, max_d)
        angle_increase = (transformed - angles) * 180 / np.pi
        ax2.plot(distances.numpy(), angle_increase.numpy(),
                label=f'Input angle={angle_deg}°', linewidth=2)

    ax2.axhline(0.0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax2.axvline(min_d, color='r', linestyle='--', alpha=0.3, linewidth=1)
    ax2.axvline(max_d, color='r', linestyle='--', alpha=0.3, linewidth=1)

    ax2.set_xlabel('Input Distance', fontsize=12)
    ax2.set_ylabel('Angle Increase (deg)', fontsize=12)
    ax2.set_title('Angle Increase vs Distance', fontsize=13)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/lfx-desktop/files/avoidance/debug/angle_trans_2d.png', dpi=150)
    print(f"已保存2D可视化到: angle_trans_2d.png")
    plt.show()


def visualize_3d_surface():
    """创建3D曲面图，显示变换后的角度作为距离和输入角度的函数"""
    print("\n" + "=" * 60)
    print("创建3D曲面可视化")
    print("=" * 60)

    min_d = 1.0
    max_d = 5.0

    # 创建网格
    distances = th.linspace(0.2, 8.0, 100)
    angles_rad = th.linspace(0, np.pi, 100)  # 0° to 180°

    D, A = th.meshgrid(distances, angles_rad, indexing='ij')

    # 计算变换后的角度
    T = transform_distance_by_angle(D, A, min_d, max_d, max_angle_increase=0.5)

    # 计算角度增加量
    angle_increase = T - A

    # 转换为numpy用于绘图
    D_np = D.numpy()
    A_deg = (A * 180 / np.pi).numpy()
    T_deg = (T * 180 / np.pi).numpy()
    increase_deg = (angle_increase * 180 / np.pi).numpy()

    # 创建双图
    fig = plt.figure(figsize=(16, 6))

    # 图1: 变换后的角度
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(D_np, A_deg, T_deg, cmap='viridis',
                             alpha=0.9, edgecolor='none')
    ax1.set_xlabel('Input Distance', fontsize=10)
    ax1.set_ylabel('Input Angle (deg)', fontsize=10)
    ax1.set_zlabel('Transformed Angle (deg)', fontsize=10)
    ax1.set_title('Transformed Angle Surface', fontsize=12, pad=20)
    ax1.set_zlim([0, 90])
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
    ax1.view_init(elev=25, azim=45)

    # 图2: 角度增加量
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(D_np, A_deg, increase_deg, cmap='plasma',
                             alpha=0.9, edgecolor='none')
    ax2.set_xlabel('Input Distance', fontsize=10)
    ax2.set_ylabel('Input Angle (deg)', fontsize=10)
    ax2.set_zlabel('Angle Increase (deg)', fontsize=10)
    ax2.set_title('Angle Increase Surface', fontsize=12, pad=20)
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    ax2.view_init(elev=25, azim=45)

    plt.tight_layout()
    plt.savefig('/home/lfx-desktop/files/avoidance/debug/angle_trans_3d.png', dpi=150)
    print(f"已保存3D可视化到: angle_trans_3d.png")
    plt.show()


def visualize_heatmap():
    """创建热力图显示角度变换行为"""
    print("\n" + "=" * 60)
    print("创建热力图可视化")
    print("=" * 60)

    min_d = 1.0
    max_d = 5.0

    # 创建网格
    distances = th.linspace(0.2, 8.0, 200)
    angles_rad = th.linspace(0, np.pi, 200)  # 0° to 180°

    D, A = th.meshgrid(distances, angles_rad, indexing='ij')

    # 计算变换后的角度
    T = transform_distance_by_angle(D, A, min_d, max_d, max_angle_increase=0.2)

    # 计算角度增加量
    angle_increase = T - A

    # 转换为numpy用于绘图
    D_np = D.numpy()
    A_deg = (A * 180 / np.pi).numpy()
    T_deg = (T * 180 / np.pi).numpy()
    increase_deg = (angle_increase * 180 / np.pi).numpy()

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 图1: 变换后角度热力图
    im1 = ax1.contourf(D_np, A_deg, T_deg, levels=50, cmap='hot')
    contour1 = ax1.contour(D_np, A_deg, T_deg, levels=10, colors='white',
                           alpha=0.3, linewidths=0.5)
    ax1.clabel(contour1, inline=True, fontsize=8, fmt='%.0f°')

    # 添加边界线
    ax1.axvline(min_d, color='cyan', linestyle='--', linewidth=2, label=f'min_d={min_d}')
    ax1.axvline(max_d, color='lime', linestyle='--', linewidth=2, label=f'max_d={max_d}')

    ax1.set_xlabel('Input Distance', fontsize=12)
    ax1.set_ylabel('Input Angle (deg)', fontsize=12)
    ax1.set_title('Transformed Angle Heatmap', fontsize=13)
    ax1.legend(loc='upper right', fontsize=10)
    fig.colorbar(im1, ax=ax1, label='Transformed Angle (deg)')

    # 图2: 角度增加量
    im2 = ax2.contourf(D_np, A_deg, increase_deg, levels=50, cmap='coolwarm')
    contour2 = ax2.contour(D_np, A_deg, increase_deg, levels=10, colors='black',
                           alpha=0.3, linewidths=0.5)
    ax2.clabel(contour2, inline=True, fontsize=8, fmt='%.0f°')

    # 添加边界线
    ax2.axvline(min_d, color='cyan', linestyle='--', linewidth=2, label=f'min_d={min_d}')
    ax2.axvline(max_d, color='lime', linestyle='--', linewidth=2, label=f'max_d={max_d}')

    ax2.set_xlabel('Input Distance', fontsize=12)
    ax2.set_ylabel('Input Angle (deg)', fontsize=12)
    ax2.set_title('Angle Increase Heatmap', fontsize=13)
    ax2.legend(loc='upper right', fontsize=10)
    fig.colorbar(im2, ax=ax2, label='Angle Increase (deg)')

    plt.tight_layout()
    plt.savefig('/home/lfx-desktop/files/avoidance/debug/angle_trans_heatmap.png', dpi=150)
    print(f"已保存热力图到: angle_trans_heatmap.png")
    plt.show()


def test_batch_operations():
    """测试批处理输入是否正常工作"""
    print("\n" + "=" * 60)
    print("测试批处理操作")
    print("=" * 60)

    min_d = 1.0
    max_d = 5.0

    # 创建随机批次
    batch_size = 10
    d = th.rand(batch_size) * 8.0  # 随机距离 0-8
    a = th.rand(batch_size) * np.pi  # 随机角度 0-180°

    result = transform_distance_by_angle(d, a, min_d, max_d, max_angle_increase=0.01)

    print(f"\n批次大小: {batch_size}")
    print(f"输入距离: {d.numpy()}")
    print(f"输入角度 (度): {(a * 180 / np.pi).numpy()}")
    print(f"输出角度 (度): {(result * 180 / np.pi).numpy()}")
    print(f"角度增加 (度): {((result - a) * 180 / np.pi).numpy()}")
    print(f"验证不超过90度: {(result <= np.pi/2).all().item()}")
    print(f"验证>=90度的角度保持不变: 检查中...")
    for i in range(batch_size):
        if a[i] >= np.pi/2:
            is_unchanged = th.isclose(result[i], a[i], atol=1e-6)
            print(f"  索引{i}: 输入={a[i]*180/np.pi:.1f}°, 输出={result[i]*180/np.pi:.1f}°, 不变={is_unchanged}")


if __name__ == "__main__":
    print("测试 transform_distance_by_angle 函数")
    print("=" * 60)

    # 运行所有测试
    test_basic_cases()
    test_batch_operations()

    # 创建可视化
    visualize_2d_slices()
    visualize_3d_surface()
    visualize_heatmap()

    print("\n" + "=" * 60)
    print("所有测试和可视化已完成！")
    print("=" * 60)

