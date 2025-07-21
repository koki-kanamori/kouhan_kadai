import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# tauchen関数
def tauchen(n, mu, rho, sigma):
    m = 3  # より標準的な値を使用
    state_space = np.linspace(mu - m*sigma, mu + m*sigma, n)
    d = (state_space[n-1] - state_space[0]) / (n-1)
    transition_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if j == 0:
                transition_matrix[i, 0] = norm.cdf((state_space[0] - rho*state_space[i] + d/2)/sigma)
            elif j == n-1:
                transition_matrix[i, n-1] = 1.0 - norm.cdf((state_space[n-1] - rho*state_space[i] - d/2)/sigma)
            else:
                transition_matrix[i, j] = norm.cdf((state_space[j] - rho*state_space[i] + d/2)/sigma) - norm.cdf((state_space[j] - rho*state_space[i] - d/2)/sigma)
    
    return transition_matrix, np.exp(state_space)

def solve_household(param, r, w, T=0):
    NA, NH = param['NA'], param['NH']
    h, a_l, a_u = param['h'], param['a_l'], param['a_u']
    sigma, beta, pi = param['sigma'], param['beta'], param['pi']
    
    a = np.linspace(a_l, a_u, NA)
    util = np.full((NA, NA, NH), -np.inf)
    
    for ia in range(NA):
        for ih in range(NH):
            for iap in range(NA):
                # 予算制約：c + a' = w*h + (1+r)*a + T
                cons = w*h[ih] + (1.0 + r)*a[ia] - a[iap] + T
                if cons > 0:
                    util[iap, ia, ih] = cons**(1.0-sigma)/(1.0-sigma)
    
    v = np.zeros((NA, NH))
    v_new = np.zeros((NA, NH))
    iaplus = np.full((NA, NH), -1)
    
    tol = 1e-6
    max_iter = 1000
    iter_count = 0
    
    while iter_count < max_iter:
        for ia in range(NA):
            for ih in range(NH):
                # 価値関数の更新
                expected_v = np.zeros(NA)
                for iap in range(NA):
                    expected_v[iap] = sum(pi[ih, ihp] * v[iap, ihp] for ihp in range(NH))
                
                reward = util[:, ia, ih] + beta * expected_v
                v_new[ia, ih] = np.max(reward)
                iaplus[ia, ih] = np.argmax(reward)
        
        if np.max(np.abs(v_new - v)) < tol:
            break
        v = v_new.copy()
        iter_count += 1
    
    if iter_count == max_iter:
        print("Warning: Maximum iterations reached")
    
    aplus = a[iaplus]
    c = w * h[np.newaxis, :] + (1.0 + r) * a[:, np.newaxis] - aplus + T
    
    return aplus, c

# パラメータ設定
param = {
    'sigma': 1.5,
    'beta': 0.98,
    'rho': 0.6,
    'sigma_eps': 0.6,
    'a_l': 0,
    'a_u': 20,
    'NA': 201,  # 計算時間短縮のため少し減らす
    'NH': 2,
    'mu_h': -0.7,
}

# 生産性グリッドと遷移確率の計算
param['pi'], param['h'] = tauchen(param['NH'], param['mu_h'], param['rho'], param['sigma_eps'])
print(f"生産性レベル: {param['h']}")
print(f"遷移確率行列:\n{param['pi']}")

# 価格設定
r, w = 0.04, 1

print("補助金導入前の家計問題を解いています...")
aplus_before, c_before = solve_household(param, r, w)

print("補助金導入後の家計問題を解いています...")
aplus_after, c_after = solve_household(param, r, w, T=1)

# 貯蓄率の計算（修正）
a = np.linspace(param['a_l'], param['a_u'], param['NA'])

# 貯蓄率 = (a' - a) / (w*h + (1+r)*a + T)
# または savings_rate = a' / income の形で計算
savings_rate_before = np.zeros((param['NA'], param['NH']))
savings_rate_after = np.zeros((param['NA'], param['NH']))

for ih in range(param['NH']):
    income_before = w * param['h'][ih] + (1 + r) * a
    income_after = w * param['h'][ih] + (1 + r) * a + 1  # T=1
    
    # 貯蓄率 = 貯蓄 / 所得
    savings_rate_before[:, ih] = aplus_before[:, ih] / income_before
    savings_rate_after[:, ih] = aplus_after[:, ih] / income_after

# グラフ描画
plt.figure(figsize=(12, 8))

# 補助金導入前
plt.plot(a, savings_rate_before[:, 0], label='Low productivity (before subsidy)', 
         color='blue', linewidth=2)
plt.plot(a, savings_rate_before[:, 1], label='High productivity (before subsidy)', 
         color='red', linewidth=2)

# 補助金導入後
plt.plot(a, savings_rate_after[:, 0], label='Low productivity (after subsidy)', 
         color='blue', linestyle='--', linewidth=2)
plt.plot(a, savings_rate_after[:, 1], label='High productivity (after subsidy)', 
         color='red', linestyle='--', linewidth=2)

plt.xlabel('Current Assets', fontsize=12)
plt.ylabel('Savings Rate', fontsize=12)
plt.title('Savings Rate vs Current Assets (Before and After Lump-Sum Subsidy T=1)', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(0, 20)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# 統計の出力
print("\n=== 結果の統計 ===")
print(f"補助金導入前の平均貯蓄率:")
print(f"  低生産性: {np.mean(savings_rate_before[:, 0]):.4f}")
print(f"  高生産性: {np.mean(savings_rate_before[:, 1]):.4f}")
print(f"補助金導入後の平均貯蓄率:")
print(f"  低生産性: {np.mean(savings_rate_after[:, 0]):.4f}")
print(f"  高生産性: {np.mean(savings_rate_after[:, 1]):.4f}")
