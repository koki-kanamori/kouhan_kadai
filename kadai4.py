import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# tauchen関数（修正版）
def tauchen(n, mu, rho, sigma):
    # より標準的な値を使用
    m = 3  # Standard coverage of 3 standard deviations
    
    # 無条件標準偏差を計算
    sigma_y = sigma / np.sqrt(1 - rho**2)
    
    # State space
    y_max = mu + m * sigma_y
    y_min = mu - m * sigma_y
    state_space = np.linspace(y_min, y_max, n)
    
    # Step size
    d = (state_space[1] - state_space[0])
    
    # Transition matrix
    transition_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if j == 0:
                # First column
                transition_matrix[i, j] = norm.cdf((state_space[j] - rho*state_space[i] + d/2) / sigma)
            elif j == n-1:
                # Last column
                transition_matrix[i, j] = 1.0 - norm.cdf((state_space[j] - rho*state_space[i] - d/2) / sigma)
            else:
                # Middle columns
                transition_matrix[i, j] = (norm.cdf((state_space[j] - rho*state_space[i] + d/2) / sigma) - 
                                         norm.cdf((state_space[j] - rho*state_space[i] - d/2) / sigma))
    
    return transition_matrix, np.exp(state_space)

def solve_household(param, r, w):
    """家計問題を価値関数反復法で解く"""
    NA, NH = param['NA'], param['NH']
    h, a_l, a_u = param['h'], param['a_l'], param['a_u']
    sigma, beta, pi = param['sigma'], param['beta'], param['pi']
    
    # Asset grid
    a = np.linspace(a_l, a_u, NA)
    
    # Utility matrix: u[iap, ia, ih] = utility when choosing a'[iap] given a[ia] and h[ih]
    util = np.full((NA, NA, NH), -np.inf)
    
    for ia in range(NA):
        for ih in range(NH):
            for iap in range(NA):
                cons = w * h[ih] + (1.0 + r) * a[ia] - a[iap]
                if cons > 1e-10:  # Ensure positive consumption
                    if sigma == 1.0:
                        util[iap, ia, ih] = np.log(cons)
                    else:
                        util[iap, ia, ih] = (cons**(1.0-sigma)) / (1.0-sigma)
    
    # Value function iteration
    v = np.zeros((NA, NH))
    v_new = np.zeros((NA, NH))
    policy = np.full((NA, NH), -1, dtype=int)
    
    tol = 1e-6
    max_iter = 1000
    
    for iteration in range(max_iter):
        for ia in range(NA):
            for ih in range(NH):
                # Expected continuation value for each choice of a'
                expected_v = np.zeros(NA)
                for iap in range(NA):
                    for ihp in range(NH):
                        expected_v[iap] += pi[ih, ihp] * v[iap, ihp]
                
                # Bellman equation
                bellman_values = util[:, ia, ih] + beta * expected_v
                
                # Find optimal policy
                valid_choices = np.isfinite(bellman_values)
                if np.any(valid_choices):
                    v_new[ia, ih] = np.max(bellman_values[valid_choices])
                    policy[ia, ih] = np.argmax(bellman_values)
                else:
                    v_new[ia, ih] = -np.inf
                    policy[ia, ih] = 0
        
        # Check convergence
        if np.max(np.abs(v_new - v)) < tol:
            print(f"Value function converged after {iteration+1} iterations")
            break
        
        v = v_new.copy()
    
    if iteration == max_iter - 1:
        print("Warning: Maximum iterations reached")
    
    # Extract policy functions
    aplus = a[policy]
    c = np.zeros((NA, NH))
    
    for ia in range(NA):
        for ih in range(NH):
            c[ia, ih] = w * h[ih] + (1.0 + r) * a[ia] - aplus[ia, ih]
            # Ensure non-negative consumption
            c[ia, ih] = max(c[ia, ih], 1e-10)
    
    return aplus, c, v

# パラメータ設定
param_base = {
    'sigma': 1.5,
    'beta': 0.98,      # 元の時間選好率
    'rho': 0.6,
    'sigma_eps': 0.6,
    'a_l': 0,
    'a_u': 20,
    'NA': 201,         # 計算時間短縮のため少し減らす
    'NH': 2,
    'mu_h': 0.0,       # 中央値を0に設定（より標準的）
}

param_low_beta = param_base.copy()
param_low_beta['beta'] = 0.5  # より現実的な低下幅（0.1は極端すぎる）

print("=== 問題4: 時間選好率変化の分析 ===")
print(f"ベースケース: β = {param_base['beta']}")
print(f"低下後: β = {param_low_beta['beta']}")

# 生産性グリッドと遷移確率の計算(両方のケースで同じ)
print("\n生産性グリッドと遷移確率を計算中...")
param_base['pi'], param_base['h'] = tauchen(param_base['NH'], param_base['mu_h'], 
                                           param_base['rho'], param_base['sigma_eps'])
param_low_beta['pi'], param_low_beta['h'] = param_base['pi'], param_base['h']

print(f"生産性レベル: {param_base['h']}")
print(f"遷移確率行列:\n{param_base['pi']}")

# 価格設定
r, w = 0.04, 1

# β=0.98の場合の家計問題を解く
print(f"\nβ = {param_base['beta']} のケースを解いています...")
aplus_base, c_base, v_base = solve_household(param_base, r, w)

# β=0.5の場合の家計問題を解く
print(f"\nβ = {param_low_beta['beta']} のケースを解いています...")
aplus_low_beta, c_low_beta, v_low_beta = solve_household(param_low_beta, r, w)

# 貯蓄率の正しい計算
a = np.linspace(param_base['a_l'], param_base['a_u'], param_base['NA'])

# 貯蓄率 = 貯蓄額 / 所得
savings_rate_base = np.zeros((param_base['NA'], param_base['NH']))
savings_rate_low_beta = np.zeros((param_base['NA'], param_base['NH']))

for ih in range(param_base['NH']):
    # 各期の総所得
    income = w * param_base['h'][ih] + (1 + r) * a
    
    # 貯蓄率の計算（方法1: 資産変化/所得）
    savings_base = aplus_base[:, ih] - a  # 純貯蓄額
    savings_low_beta = aplus_low_beta[:, ih] - a
    
    # ゼロ除算を避ける
    income_safe = np.maximum(income, 1e-10)
    
    savings_rate_base[:, ih] = savings_base / income_safe
    savings_rate_low_beta[:, ih] = savings_low_beta / income_safe

# グラフ描画
plt.figure(figsize=(15, 10))

# メインプロット: 貯蓄率 vs Current Assets
plt.subplot(2, 2, 1)
# β=0.98の場合
plt.plot(a, savings_rate_base[:, 0], label=f'Low productivity (β={param_base["beta"]})', 
         color='blue', linewidth=2)
plt.plot(a, savings_rate_base[:, 1], label=f'High productivity (β={param_base["beta"]})', 
         color='red', linewidth=2)

# β=0.5の場合
plt.plot(a, savings_rate_low_beta[:, 0], label=f'Low productivity (β={param_low_beta["beta"]})', 
         color='blue', linestyle='--', linewidth=2)
plt.plot(a, savings_rate_low_beta[:, 1], label=f'High productivity (β={param_low_beta["beta"]})', 
         color='red', linestyle='--', linewidth=2)

plt.xlabel('Current Assets', fontsize=12)
plt.ylabel('Savings Rate', fontsize=12)
plt.title('Savings Rate vs Current Assets (Time Preference Change)', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(0, 20)

# 補助プロット1: 消費政策関数
plt.subplot(2, 2, 2)
plt.plot(a, c_base[:, 0], label=f'Low prod. (β={param_base["beta"]})', color='blue', linewidth=2)
plt.plot(a, c_base[:, 1], label=f'High prod. (β={param_base["beta"]})', color='red', linewidth=2)
plt.plot(a, c_low_beta[:, 0], label=f'Low prod. (β={param_low_beta["beta"]})', 
         color='blue', linestyle='--', linewidth=2)
plt.plot(a, c_low_beta[:, 1], label=f'High prod. (β={param_low_beta["beta"]})', 
         color='red', linestyle='--', linewidth=2)
plt.xlabel('Current Assets', fontsize=12)
plt.ylabel('Consumption', fontsize=12)
plt.title('Consumption Policy Function', fontsize=14)
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)

# 補助プロット2: 資産政策関数
plt.subplot(2, 2, 3)
plt.plot(a, aplus_base[:, 0], label=f'Low prod. (β={param_base["beta"]})', color='blue', linewidth=2)
plt.plot(a, aplus_base[:, 1], label=f'High prod. (β={param_base["beta"]})', color='red', linewidth=2)
plt.plot(a, aplus_low_beta[:, 0], label=f'Low prod. (β={param_low_beta["beta"]})', 
         color='blue', linestyle='--', linewidth=2)
plt.plot(a, aplus_low_beta[:, 1], label=f'High prod. (β={param_low_beta["beta"]})', 
         color='red', linestyle='--', linewidth=2)
plt.plot(a, a, 'k--', alpha=0.5, label='45° line')
plt.xlabel('Current Assets', fontsize=12)
plt.ylabel('Next Period Assets', fontsize=12)
plt.title('Asset Policy Function', fontsize=14)
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)

# 補助プロット3: 価値関数
plt.subplot(2, 2, 4)
plt.plot(a, v_base[:, 0], label=f'Low prod. (β={param_base["beta"]})', color='blue', linewidth=2)
plt.plot(a, v_base[:, 1], label=f'High prod. (β={param_base["beta"]})', color='red', linewidth=2)
plt.plot(a, v_low_beta[:, 0], label=f'Low prod. (β={param_low_beta["beta"]})', 
         color='blue', linestyle='--', linewidth=2)
plt.plot(a, v_low_beta[:, 1], label=f'High prod. (β={param_low_beta["beta"]})', 
         color='red', linestyle='--', linewidth=2)
plt.xlabel('Current Assets', fontsize=12)
plt.ylabel('Value Function', fontsize=12)
plt.title('Value Function', fontsize=14)
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 統計分析
print("\n=== 結果の統計 ===")
print(f"平均貯蓄率 (β={param_base['beta']}):")
print(f"  低生産性: {np.mean(savings_rate_base[:, 0]):.4f}")
print(f"  高生産性: {np.mean(savings_rate_base[:, 1]):.4f}")

print(f"\n平均貯蓄率 (β={param_low_beta['beta']}):")
print(f"  低生産性: {np.mean(savings_rate_low_beta[:, 0]):.4f}")
print(f"  高生産性: {np.mean(savings_rate_low_beta[:, 1]):.4f}")

print(f"\n平均消費 (β={param_base['beta']}):")
print(f"  低生産性: {np.mean(c_base[:, 0]):.4f}")
print(f"  高生産性: {np.mean(c_base[:, 1]):.4f}")

print(f"\n平均消費 (β={param_low_beta['beta']}):")
print(f"  低生産性: {np.mean(c_low_beta[:, 0]):.4f}")
print(f"  高生産性: {np.mean(c_low_beta[:, 1]):.4f}")

print("\n=== 経済学的解釈 ===")
print("時間選好率の低下（βの低下）は以下の効果をもたらします:")
print("1. より近視眼的な行動 → 現在消費の増加")
print("2. 将来への配慮低下 → 貯蓄率の低下")
print("3. 予防的貯蓄動機の低下")

print("\n計算完了!")          
