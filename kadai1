import numpy as np
import matplotlib.pyplot as plt

# パラメータ設定
param = {
    'sigma': 2.0,                    # 相対的リスク回避度
    'beta': 0.985**20,               # 割引因子（20年×1期間）
    'a_l': 0.0,                      # 最小資産
    'a_u': 20.0,                     # 最大資産
    'NA': 401,                       # 資産グリッド数
    'NH': 3,                         # 生産性タイプ数
}

# 生産性と遷移確率（設問で与えられている）
h = np.array([0.8027, 1.0, 1.2457])
P = np.array([
    [0.7451, 0.2528, 0.0021],
    [0.1360, 0.7281, 0.1360],
    [0.0021, 0.2528, 0.7451]
])

# 利子率と賃金
r = 0.0252
w = 1.0

# ライフサイクルモデル（3期間）の解法
def solve_lifecycle(param, r, w, h, P):
    NA, NH = param['NA'], param['NH']
    sigma, beta = param['sigma'], param['beta']
    
    a_grid = np.linspace(param['a_l'], param['a_u'], NA)
    
    # ステージ3（老年期）：消費のみ
    V3 = np.zeros((NA,))
    
    # ステージ2（中年期）：生産性ごとに最適化
    V2 = np.zeros((NA, NH))
    policy2 = np.zeros((NA, NH))  # a2'

    for ih in range(NH):
        for ia in range(NA):
            max_val = -1e10
            for iap in range(NA):
                cons = w * h[ih] + (1 + r) * a_grid[ia] - a_grid[iap]
                if cons > 0:
                    util = cons**(1 - sigma) / (1 - sigma)
                    val = util + beta * V3[iap]
                    if val > max_val:
                        max_val = val
                        policy2[ia, ih] = a_grid[iap]
            V2[ia, ih] = max_val

    # ステージ1（若年期）：生産性ごとに最適化
    V1 = np.zeros((NA, NH))
    policy1 = np.zeros((NA, NH))  # a1'

    for ih in range(NH):
        for ia in range(NA):
            max_val = -1e10
            for iap in range(NA):
                cons = w * h[ih] + (1 + r) * a_grid[ia] - a_grid[iap]
                if cons > 0:
                    EV2 = np.dot(P[ih], V2[iap, :])
                    util = cons**(1 - sigma) / (1 - sigma)
                    val = util + beta * EV2
                    if val > max_val:
                        max_val = val
                        policy1[ia, ih] = a_grid[iap]
            V1[ia, ih] = max_val

    return policy1, a_grid

# 実行
policy1, a_grid = solve_lifecycle(param, r, w, h, P)

# 描画
plt.figure(figsize=(10, 6))
labels = ['Low productivity', 'Mid productivity', 'High productivity']
colors = ['blue', 'green', 'red']
for i in range(3):
    plt.plot(a_grid, policy1[:, i], label=labels[i], color=colors[i])
plt.plot(a_grid, a_grid, '--', color='gray', label="45° line")
plt.xlabel("Current Asset $a$")
plt.ylabel("Next Period Asset $a'$")
plt.title("Savings Policy Function by Productivity Type (No Pension)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
