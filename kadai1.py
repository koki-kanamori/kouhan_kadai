import numpy as np
import matplotlib.pyplot as plt

# 問題1のパラメータ設定
gamma = 2.0
beta = 0.985

# 確率行列P（状態遷移確率）
P = np.array([
    [0.7451, 0.2528, 0.0021],
    [0.1360, 0.7281, 0.1360],
    [0.0021, 0.2528, 0.7451]
])

# 所得状態（文書から推定）
income_states = np.array([0.8027, 1.0, 1.2457])

# 初期資産
initial_wealth = 20.0

# 利子率（文書から1.025が見えるので）
r = 0.025

print("=== 問題1: 3期間消費最適化問題 ===")
print(f"パラメータ: γ = {gamma}, β = {beta}")
print(f"初期資産: {initial_wealth}")
print(f"利子率: {r}")
print("\n所得状態:")
for i, income in enumerate(income_states):
    print(f"  状態{i+1}: {income:.4f}")

print("\n遷移確率行列P:")
print(P)

def utility(c, gamma):
    """効用関数 u(c) = c^(1-γ)/(1-γ)"""
    if gamma == 1.0:
        return np.log(c)
    else:
        return (c**(1-gamma)) / (1-gamma)

def solve_three_period_problem():
    """3期間消費最適化問題を後ろ向き帰納法で解く"""
    
    n_states = len(income_states)
    
    # 資産グリッド（各期間で選択可能な資産水準）
    n_assets = 101
    max_assets = initial_wealth * (1 + r)**2 + sum(income_states) * 2
    asset_grid = np.linspace(0, max_assets, n_assets)
    
    # 価値関数と政策関数の初期化
    V = np.zeros((3, n_assets, n_states))  # V[t, a, s]
    C = np.zeros((3, n_assets, n_states))  # C[t, a, s]
    A_next = np.zeros((2, n_assets, n_states))  # A_next[t, a, s] for t=0,1
    
    print("\n=== 後ろ向き帰納法による解法 ===")
    
    # 第3期（最終期）t=2
    print("第3期の最適化...")
    for ia, assets in enumerate(asset_grid):
        for s in range(n_states):
            # 最終期では全ての資産を消費
            total_resources = assets * (1 + r) + income_states[s]
            if total_resources > 0:
                C[2, ia, s] = total_resources
                V[2, ia, s] = utility(total_resources, gamma)
                A_next = np.zeros((2, n_assets, n_states))  # 最終期なので次期資産は0
            else:
                C[2, ia, s] = 1e-10
                V[2, ia, s] = -np.inf
    
    # 第2期 t=1
    print("第2期の最適化...")
    for ia, assets in enumerate(asset_grid):
        for s in range(n_states):
            total_resources = assets * (1 + r) + income_states[s]
            
            best_value = -np.inf
            best_consumption = 0
            best_assets_next = 0
            
            # 可能な次期資産水準を試行
            for ia_next, assets_next in enumerate(asset_grid):
                consumption = total_resources - assets_next
                
                if consumption > 1e-10:  # 正の消費
                    # 期待効用の計算
                    expected_value = 0
                    for s_next in range(n_states):
                        expected_value += P[s, s_next] * V[2, ia_next, s_next]
                    
                    # ベルマン方程式
                    total_value = utility(consumption, gamma) + beta * expected_value
                    
                    if total_value > best_value:
                        best_value = total_value
                        best_consumption = consumption
                        best_assets_next = assets_next
            
            V[1, ia, s] = best_value
            C[1, ia, s] = best_consumption
            A_next[1, ia, s] = best_assets_next
    
    # 第1期 t=0
    print("第1期の最適化...")
    for ia, assets in enumerate(asset_grid):
        for s in range(n_states):
            total_resources = assets * (1 + r) + income_states[s]
            
            best_value = -np.inf
            best_consumption = 0
            best_assets_next = 0
            
            # 可能な次期資産水準を試行
            for ia_next, assets_next in enumerate(asset_grid):
                consumption = total_resources - assets_next
                
                if consumption > 1e-10:  # 正の消費
                    # 期待効用の計算
                    expected_value = 0
                    for s_next in range(n_states):
                        expected_value += P[s, s_next] * V[1, ia_next, s_next]
                    
                    # ベルマン方程式
                    total_value = utility(consumption, gamma) + beta * expected_value
                    
                    if total_value > best_value:
                        best_value = total_value
                        best_consumption = consumption
                        best_assets_next = assets_next
            
            V[0, ia, s] = best_value
            C[0, ia, s] = best_consumption
            A_next[0, ia, s] = best_assets_next
    
    return V, C, A_next, asset_grid

# 問題を解く
V, C, A_next, asset_grid = solve_three_period_problem()

# 貯蓄率の計算
def calculate_savings_rates(C, A_next, asset_grid, income_states, r):
    """貯蓄率を計算する関数"""
    n_assets, n_states = C.shape[1], C.shape[2]
    savings_rates = np.zeros((2, n_assets, n_states))  # 第1期と第2期のみ
    
    for t in range(2):  # t=0,1（第3期は最終期なので貯蓄なし）
        for ia in range(n_assets):
            for s in range(n_states):
                # 今期の総所得
                current_assets = asset_grid[ia]
                total_income = current_assets * (1 + r) + income_states[s]
                
                # 貯蓄額 = 次期資産
                savings = A_next[t, ia, s]
                
                # 貯蓄率 = 貯蓄額 / 総所得
                if total_income > 1e-10:
                    savings_rates[t, ia, s] = savings / total_income
                else:
                    savings_rates[t, ia, s] = 0
    
    return savings_rates

# 貯蓄率を計算
savings_rates = calculate_savings_rates(C, A_next, asset_grid, income_states, r)

# グラフ作成: 貯蓄率 vs Current Assets
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Savings Rate vs Current Assets (3-Period Problem)', fontsize=16)

colors = ['blue', 'red', 'green']
state_labels = ['Low income', 'Medium income', 'High income']

# 第1期の貯蓄率
ax = axes[0]
for s in range(len(income_states)):
    # 資産レベルが高すぎる部分は除外（現実的でない値）
    valid_assets = asset_grid <= 30  # 資産30まで表示
    ax.plot(asset_grid[valid_assets], savings_rates[0, valid_assets, s], 
            color=colors[s], linewidth=2, 
            label=f'{state_labels[s]} (y={income_states[s]:.3f})')

ax.set_xlabel('Current Assets', fontsize=12)
ax.set_ylabel('Savings Rate', fontsize=12)
ax.set_title('Period 1 Savings Rate', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 30)
ax.set_ylim(0, 1)

# 第2期の貯蓄率
ax = axes[1]
for s in range(len(income_states)):
    valid_assets = asset_grid <= 30
    ax.plot(asset_grid[valid_assets], savings_rates[1, valid_assets, s], 
            color=colors[s], linewidth=2,
            label=f'{state_labels[s]} (y={income_states[s]:.3f})')

ax.set_xlabel('Current Assets', fontsize=12)
ax.set_ylabel('Savings Rate', fontsize=12)
ax.set_title('Period 2 Savings Rate', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 30)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.show()

# 統計分析
print(f"\n=== 貯蓄率の分析 ===")
for t in range(2):
    print(f"\n第{t+1}期:")
    for s in range(len(income_states)):
        # 資産レベル0-20の範囲での平均貯蓄率
        valid_range = (asset_grid >= 0) & (asset_grid <= 20)
        avg_savings_rate = np.mean(savings_rates[t, valid_range, s])
        print(f"  {state_labels[s]}: 平均貯蓄率 = {avg_savings_rate:.4f}")

# 初期資産20での詳細分析
initial_asset_index = np.argmin(np.abs(asset_grid - initial_wealth))
actual_initial_assets = asset_grid[initial_asset_index]

print(f"\n=== 初期資産{actual_initial_assets:.1f}での貯蓄率 ===")
for s in range(len(income_states)):
    sr1 = savings_rates[0, initial_asset_index, s]
    print(f"{state_labels[s]}: 第1期貯蓄率 = {sr1:.4f}")
    
    # 第2期の期待貯蓄率
    a2 = A_next[0, initial_asset_index, s]
    a2_index = np.argmin(np.abs(asset_grid - a2))
    expected_sr2 = 0
    for s2 in range(len(income_states)):
        prob = P[s, s2]
        sr2 = savings_rates[1, a2_index, s2]
        expected_sr2 += prob * sr2
    print(f"{state_labels[s]}: 第2期期待貯蓄率 = {expected_sr2:.4f}")

print("\n計算完了!")
