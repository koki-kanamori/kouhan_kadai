import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def tauchen(n, mu, rho, sigma):
    """
    Tauchen method for discretizing AR(1) process
    """
    # Coverage parameter
    m = 3  # Standard coverage of 3 standard deviations
    
    # State space
    sigma_y = sigma / np.sqrt(1 - rho**2)  # Unconditional standard deviation
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
    """
    Solve household problem using value function iteration
    """
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
    
    # Extract policy functions
    aplus = a[policy]
    c = np.zeros((NA, NH))
    
    for ia in range(NA):
        for ih in range(NH):
            c[ia, ih] = w * h[ih] + (1.0 + r) * a[ia] - aplus[ia, ih]
    
    return aplus, c, v, a

# パラメータ設定
param = {
    'sigma': 1.5,           # Risk aversion parameter
    'beta': 0.98,           # Discount factor
    'rho': 0.6,             # Persistence of productivity shock
    'sigma_eps': 0.6,       # Standard deviation of productivity shock
    'a_l': 0.0,             # Lower bound on assets
    'a_u': 20.0,            # Upper bound on assets
    'NA': 201,              # Number of asset grid points
    'NH': 2,                # Number of productivity states
    'mu_h': 0.0,            # Mean of log productivity (centered at 0)
}

print("Computing productivity grid and transition matrix...")
# 生産性グリッドと遷移確率の計算
param['pi'], param['h'] = tauchen(param['NH'], param['mu_h'], param['rho'], param['sigma_eps'])

print("Productivity states:", param['h'])
print("Transition matrix:")
print(param['pi'])

# 価格設定
r, w = 0.04, 1.0

print("\nSolving household problem...")
# 家計問題を解く
aplus, c, v, a = solve_household(param, r, w)

# 貯蓄率の計算（修正版）
def calculate_savings_rate(a, aplus, h, r, w):
    """
    Calculate savings rate correctly
    """
    savings_rate = np.zeros_like(aplus)
    
    for ih in range(len(h)):
        # Total income = labor income + capital income
        total_income = w * h[ih] + (1 + r) * a
        
        # Savings rate = next period assets / total income
        # (正確には貯蓄額/所得だが、ここでは資産蓄積率として解釈)
        savings_rate[:, ih] = aplus[:, ih] / total_income
    
    return savings_rate

# 貯蓄率を計算
savings_rate = calculate_savings_rate(a, aplus, param['h'], r, w)

# Results verification
print(f"\nAsset grid range: [{a[0]:.2f}, {a[-1]:.2f}]")
print(f"Consumption range: [{np.min(c[c>0]):.3f}, {np.max(c):.3f}]")
print(f"Savings range: [{np.min(aplus):.3f}, {np.max(aplus):.3f}]")
print(f"Savings rate range: [{np.min(savings_rate):.3f}, {np.max(savings_rate):.3f}]")

# メインのグラフ: 貯蓄率 vs Current Assets
plt.figure(figsize=(12, 8))

# 低生産性と高生産性の貯蓄率をプロット
plt.plot(a, savings_rate[:, 0], 
         label=f'Low productivity (h={param["h"][0]:.3f})', 
         linewidth=3, color='blue')
plt.plot(a, savings_rate[:, 1], 
         label=f'High productivity (h={param["h"][1]:.3f})', 
         linewidth=3, color='red')

plt.xlabel('Current Assets', fontsize=14)
plt.ylabel('Savings Rate', fontsize=14)
plt.title('Savings Rate vs Current Assets (Infinite Horizon Problem)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(0, 20)
plt.ylim(0, 1)

# より見やすいスタイル設定
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# 追加の分析用グラフ
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Infinite Horizon Household Problem Analysis', fontsize=16)

# Plot 1: Asset Policy Function
ax = axes[0, 0]
ax.plot(a, aplus[:, 0], label='Low productivity', linewidth=2, color='blue')
ax.plot(a, aplus[:, 1], label='High productivity', linewidth=2, color='red')
ax.plot(a, a, 'k--', alpha=0.5, label='45° line')
ax.set_xlabel('Current Assets')
ax.set_ylabel('Next Period Assets')
ax.set_title('Asset Policy Function')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Consumption Policy
ax = axes[0, 1]
ax.plot(a, c[:, 0], label='Low productivity', linewidth=2, color='blue')
ax.plot(a, c[:, 1], label='High productivity', linewidth=2, color='red')
ax.set_xlabel('Current Assets')
ax.set_ylabel('Consumption')
ax.set_title('Consumption Policy Function')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Savings Rate (same as main plot)
ax = axes[1, 0]
ax.plot(a, savings_rate[:, 0], label='Low productivity', linewidth=2, color='blue')
ax.plot(a, savings_rate[:, 1], label='High productivity', linewidth=2, color='red')
ax.set_xlabel('Current Assets')
ax.set_ylabel('Savings Rate')
ax.set_title('Savings Rate')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Value Function
ax = axes[1, 1]
ax.plot(a, v[:, 0], label='Low productivity', linewidth=2, color='blue')
ax.plot(a, v[:, 1], label='High productivity', linewidth=2, color='red')
ax.set_xlabel('Current Assets')
ax.set_ylabel('Value Function')
ax.set_title('Value Function')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 詳細分析
print("\n=== Detailed Analysis ===")

# 異なる資産レベルでの貯蓄率
asset_levels = [0, 5, 10, 15, 20]
print("\nSavings rates at different asset levels:")
print("Assets\tLow Prod.\tHigh Prod.")
print("-" * 35)
for asset_level in asset_levels:
    idx = np.argmin(np.abs(a - asset_level))
    actual_asset = a[idx]
    sr_low = savings_rate[idx, 0]
    sr_high = savings_rate[idx, 1]
    print(f"{actual_asset:.1f}\t{sr_low:.4f}\t\t{sr_high:.4f}")

# 平均統計
print(f"\nAverage savings rates:")
print(f"Low productivity: {np.mean(savings_rate[:, 0]):.4f}")
print(f"High productivity: {np.mean(savings_rate[:, 1]):.4f}")

# 定常状態の近似分析（高資産レベルでの行動）
high_asset_idx = a >= 15
print(f"\nSteady-state approximation (assets >= 15):")
print(f"Low productivity: {np.mean(savings_rate[high_asset_idx, 0]):.4f}")
print(f"High productivity: {np.mean(savings_rate[high_asset_idx, 1]):.4f}")

# 生産性ショックの影響
print(f"\nProductivity shock effects:")
print(f"Productivity ratio (high/low): {param['h'][1]/param['h'][0]:.3f}")
print(f"Average savings rate ratio (high/low): {np.mean(savings_rate[:, 1])/np.mean(savings_rate[:, 0]):.3f}")

print("\nAnalysis completed!")
