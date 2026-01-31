import torch as th
import numpy as np

def transform_distance_by_angle(d, a, min_d, max_d, max_angle_increase=0.8):
    d = th.as_tensor(d, dtype=th.float32)
    a = th.as_tensor(a, dtype=th.float32)

    pi_2 = th.pi / 2  # 90度

    # 条件1: 角度 >= 90度 -> 不做变换
    angle_mask = a >= pi_2

    # 条件2: 距离 >= max_d -> 不做变换
    distance_mask = d >= max_d

    # 不变换的情况
    no_transform_mask = angle_mask | distance_mask

    print(f"angle_mask: {angle_mask}")
    print(f"distance_mask: {distance_mask}")
    print(f"no_transform_mask: {no_transform_mask}")

    # 计算距离因子
    d_clamped = d.clamp(min=min_d, max=max_d)
    d_normalized = (d_clamped - min_d) / (max_d - min_d)
    d_factor = (1.0 - d_normalized) ** 2

    # 对于 d < min_d 的情况，使用最大因子
    d_factor = th.where(d < min_d, th.ones_like(d_factor), d_factor)

    # 计算角度增量
    angle_gap = pi_2 - a  # 到90度的距离
    print(f"angle_gap: {angle_gap * 180 / np.pi} degrees")

    angle_increase = angle_gap * (1.0 - th.exp(-3.0 * d_factor * max_angle_increase))
    print(f"angle_increase: {angle_increase * 180 / np.pi} degrees")

    # 计算变换后的角度
    a_transformed = th.where(
        no_transform_mask,
        a,  # 不变换
        a + angle_increase  # 增加角度
    )

    print(f"before clamp: {a_transformed * 180 / np.pi} degrees")

    # 确保不超过90度（安全检查）
    a_transformed = th.clamp(a_transformed, max=pi_2)

    print(f"after clamp: {a_transformed * 180 / np.pi} degrees")

    return a_transformed

# 测试90度以上的角度
min_d = 0.2
max_d = 4.0
d = th.tensor([0.5, 1.5, 2.5, 3.5])
a = th.tensor([95, 120, 150, 175]) * np.pi / 180

print('输入角度(度):', (a * 180 / np.pi).numpy())
print('输入距离:', d.numpy())
print()

result = transform_distance_by_angle(d, a, min_d, max_d)

print()
print('输出角度(度):', (result * 180 / np.pi).numpy())
print('角度变化(度):', ((result - a) * 180 / np.pi).numpy())

