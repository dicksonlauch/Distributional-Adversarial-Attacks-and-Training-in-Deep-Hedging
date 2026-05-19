import torch
import torch.nn as nn
import numpy as np
from scipy import integrate, stats
from Heston_util import PathGeneratorHeston

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# ============================================================================
# 1. PATH GENERATOR — Extends Heston with VIX spot + VIX Futures (quadrature)
# ============================================================================

class VIXHestonPathGenerator(PathGeneratorHeston):
    """
    Extends the Heston path generator to also produce VIX spot and
    VIX futures prices (front-month F1) at each timestep.

    VIX_t = 100 * sqrt(a_vix * v_t + b_vix)
    where a_vix = (1 - exp(-kappa * tau_vix)) / (kappa * tau_vix)
          b_vix = theta * (1 - a_vix)

    F(tau_fut, v_t) = 100 * E[sqrt(a_vix * V_{t+tau_fut} + b_vix)]
    computed via numerical quadrature over the non-central chi-squared
    distribution of V_{t+tau_fut} | v_t.

    Fixed tenors: F1 has tau_fut1=30/365.
    """
    def __init__(self, s0, v0, alpha, b, sigma, rho, timestep, T,
                 tau_vix=30/365, tau_fut1=30/365):
        super().__init__(s0, v0, alpha, b, sigma, rho, timestep, T)
        self.tau_vix = tau_vix
        self.tau_fut1 = tau_fut1

        # VIX formula coefficients: VIX = 100 * sqrt(a_vix * v + b_vix)
        self.a_vix = (1 - np.exp(-self.alpha * self.tau_vix)) / (self.alpha * self.tau_vix)
        self.b_vix = self.b * (1 - self.a_vix)

        # Precompute lookup tables for VIX futures pricing
        self._precompute_futures_tables()

    def _vix_futures_price_scalar(self, v_t, tau_fut):
        """
        Compute VIX futures price F(tau_fut) for a single v_t value
        using numerical quadrature.

        V_{t+tau_fut} = c * Y, where Y ~ ncx2(d, lambda)
        F = 100 * E[sqrt(a_vix * c * Y + b_vix)]
        """
        kappa, theta, sigma_v = self.alpha, self.b, self.sigma

        c = sigma_v**2 * (1 - np.exp(-kappa * tau_fut)) / (4 * kappa)
        d = 4 * kappa * theta / sigma_v**2
        lam = v_t * np.exp(-kappa * tau_fut) / c

        def integrand(y):
            pdf_val = stats.ncx2.pdf(y, d, lam)
            return np.sqrt(self.a_vix * c * y + self.b_vix) * pdf_val

        result, _ = integrate.quad(integrand, 0, np.inf, limit=200)
        return 100 * result

    def _precompute_futures_tables(self, n_grid=1000):
        """
        Precompute VIX futures prices on a grid of v values, so we can
        use fast interpolation instead of per-sample quadrature.
        """
        self.v_grid = np.linspace(1e-4, 3.0, n_grid)

        print("Precomputing VIX futures lookup table (F1)...")
        self.F1_grid = np.array([
            self._vix_futures_price_scalar(v, self.tau_fut1) for v in self.v_grid
        ])
        print("VIX futures lookup tables ready.")

    def compute_vix_spot(self, V):
        """
        Compute VIX spot from variance paths.
        V: numpy array or torch tensor of shape (N, T+1)
        Returns: same shape
        """
        if isinstance(V, torch.Tensor):
            return 100 * torch.sqrt(self.a_vix * V + self.b_vix)
        return 100 * np.sqrt(self.a_vix * V + self.b_vix)

    def compute_vix_futures(self, V):
        """
        Compute VIX futures prices via interpolation on precomputed table.
        V: numpy array of shape (N, T+1)
        Returns: F1 of same shape
        """
        V_np = V if isinstance(V, np.ndarray) else V.numpy()
        F1 = np.interp(V_np.ravel(), self.v_grid, self.F1_grid).reshape(V_np.shape)
        return F1

    def generate(self, sample_size):
        """
        Generate full dataset: (S, V, VIX, F1).
        All tensors have shape (sample_size, timestep+1).
        """
        S, V = self.generate_paths(sample_size)
        VIX = self.compute_vix_spot(V)
        F1 = self.compute_vix_futures(V)
        return (torch.Tensor(S), torch.Tensor(V),
                torch.Tensor(VIX), torch.Tensor(F1))


# ============================================================================
# 2. FEATURE CONSTRUCTION
# ============================================================================

def construct_features(S, VIX, F1, K, T, dt, sequence_length):
    """
    Construct the input feature tensor from raw price paths.

    Features per timestep t (for t = 0..sequence_length-1):
        0: ln(S_t / K)           — moneyness
        1: ln(S_t / S_{t-1})     — equity log-return (0 at t=0)
        2: ln(VIX_t / VIX_{t-1}) — VIX log-return (0 at t=0)
        3: ln(F1_t / F1_{t-1})   — front-month return (0 at t=0)
        4: tau_opt               — remaining time to option maturity
        5: tau_fut1              — time to futures maturity (constant for fixed tenor)

    Args:
        S, VIX, F1: tensors of shape (N, T+1)
        K: strike price
        T: total time to maturity
        dt: time step size
        sequence_length: number of hedging steps

    Returns:
        features: tensor of shape (N, sequence_length, 6)
    """
    N = S.shape[0]

    # Use prices at t=0..T-1 (we hedge at these times)
    S_hedge = S[:, :-1]       # (N, T)
    VIX_hedge = VIX[:, :-1]
    F1_hedge = F1[:, :-1]

    # Feature 0: moneyness
    moneyness = torch.log(S_hedge / K)  # (N, T)

    # Features 1-3: log-returns (0 at t=0)
    log_ret_S = torch.zeros_like(S_hedge)
    log_ret_S[:, 1:] = torch.log(S_hedge[:, 1:] / S_hedge[:, :-1])

    log_ret_VIX = torch.zeros_like(VIX_hedge)
    log_ret_VIX[:, 1:] = torch.log(VIX_hedge[:, 1:] / VIX_hedge[:, :-1])

    log_ret_F1 = torch.zeros_like(F1_hedge)
    log_ret_F1[:, 1:] = torch.log(F1_hedge[:, 1:] / F1_hedge[:, :-1])

    # Feature 4: remaining time to option maturity
    tau_opt = (T - dt * torch.arange(sequence_length, device=S.device)).unsqueeze(0).expand(N, -1)

    # Feature 5: time to futures maturity (constant for fixed tenor)
    tau_fut = torch.full((N, sequence_length), 30/365, device=S.device)

    # Stack: (N, T, 6)
    features = torch.stack([
        moneyness, log_ret_S, log_ret_VIX, log_ret_F1,
        tau_opt, tau_fut
    ], dim=-1)

    return features


# ============================================================================
# 3. NETWORK ARCHITECTURE
# ============================================================================
class RNN_BN_simple_VIX(nn.Module):
    """
    Time-unrolled feedforward network for VIX hedging WITH RECURRENCE.
    Input:  6 features + 2 previous holdings = 8
    Output: 2 holdings per timestep [delta_S, delta_F1]
    """
    def __init__(self, sequence_length, input_size=6, output_size=2, hidden_size=64,
                 device=torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")):
        super(RNN_BN_simple_VIX, self).__init__()
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        
        # 1. Normalize ONLY the market features (the 6 original features)
        self.bn_feats = nn.ModuleList([nn.BatchNorm1d(input_size) for _ in range(sequence_length)])
        
        # 2. Feedforward layers (takes normalized features + UNNORMALIZED memory)
        # Note: The first layer now takes input_size + output_size (6 + 2 = 8)
        self.rnn = nn.ModuleList([nn.Sequential(
            nn.Linear(input_size + output_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        ) for _ in range(sequence_length)])
        
        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.eps = 1e-3
                module.momentum = 0.3

    def forward(self, input):
        """
        Args:
            input: (N, sequence_length, input_size)
        Returns:
            output: (N, sequence_length, output_size)
        """
        input = input.transpose(0, 1)  # (T, N, input_size)
        assert input.shape[0] == self.sequence_length, 'input shape is not correct'
        
        outputs = []
        # Initialize the previous holding as zeros for t=0. 
        # Inherit device from input to prevent MPS/CPU crashes
        prev_output = torch.zeros(input.shape[1], self.output_size, device=input.device)
        
        for i in range(self.sequence_length):
            # Normalize market data
            norm_x = self.bn_feats[i](input[i])
            
            # Combine normalized market features with the agent's absolute previous holdings
            input_hidden = torch.cat([norm_x, prev_output], dim=1)
            
            output_i = self.rnn[i](input_hidden)
            outputs.append(output_i)
            
            # Update the memory for the next timestep
            prev_output = output_i
            
        return torch.stack(outputs, dim=0).transpose(0, 1)  # (N, T, output_size)
# ============================================================================
# 4. LOSS FUNCTION — CVaR with 3 instruments (S, F1, F2)
# ============================================================================

def find_optimal_p0(X_total, loss_alpha=0.5, tolerance=1e-4):
    return X_total.quantile(loss_alpha).item()


class loss_Entropic_VIX(nn.Module):
    """
    Entropic Risk (Exponential Utility) Loss for VIX Deep Hedging.
    Penalizes portfolio variance to enforce pure liability replication.
    """
    def __init__(self, Strike_price, risk_aversion=0.01, trans_cost_rate=0.):
        super(loss_Entropic_VIX, self).__init__()
        self.K = Strike_price
        self.lambd = risk_aversion
        self.transaction_cost_rate = trans_cost_rate

    def terminal_payoff(self, final_price):
        return torch.max(final_price - self.K, torch.zeros_like(final_price))

    def forward(self, holding, S, F1, VIX, output_hedging_error=False, p0=None):
        # p0 is accepted purely for compatibility with your old training loops, 
        # but Entropic Risk does not use a VaR threshold.
        
        # 1. Price changes
        delta_S = S[:, 1:] - S[:, :-1]      
        
        # 2. Daily Roll Yield (Theta) to prevent constant-maturity arbitrage
        daily_roll_yield = (VIX[:, :-1] - F1[:, :-1]) * (1.0 / 30.0)
        true_delta_F1 = (F1[:, 1:] - F1[:, :-1]) + daily_roll_yield

        # Stack: (N, T, 2)
        delta_price = torch.stack([delta_S, true_delta_F1], dim=-1)

        # 3. PnL = sum over time and instruments
        PnL = (holding * delta_price).sum(dim=(1, 2))  # (N,)

        # 4. Option Liability & Hedging Error (X)
        C_T = self.terminal_payoff(S[:, -1])
        X = C_T - PnL

        # 5. Transaction costs (with 5x multiplier for VIX futures)
        if self.transaction_cost_rate > 0:
            price_levels = torch.stack([S[:, :-1], F1[:, :-1] * 5.0], dim=-1)
            delta_holding = torch.diff(holding, dim=1, prepend=holding[:, :1])
            tc = (delta_holding.abs() * self.transaction_cost_rate * price_levels).sum(dim=(1, 2))
            X = X + tc

        # 6. Entropic Risk Measure: (1/lambda) * ln( E[ exp(lambda * X) ] )
        # Numerical stability trick: factor out the max value to prevent float32 overflow
        max_X = torch.max(X).detach()
        loss = max_X + (1.0 / self.lambd) * torch.log(torch.mean(torch.exp(self.lambd * (X - max_X))))

        if output_hedging_error:
            return loss, X.mean()
        return loss

class loss_CVAR_VIX(nn.Module):
    """
    CVaR loss function for hedging with 2 instruments: equity (S),
    front-month VIX future (F1).

    PnL = sum_t [delta_S * dS + delta_F1 * dF1]
    X = C_T - PnL + transaction_costs
    Loss = CVaR_alpha(X)
    """
    def __init__(self, Strike_price, alpha_loss=0.5,
                 p0_mode='search', trans_cost_rate=0.):
        super(loss_CVAR_VIX, self).__init__()
        self.K = Strike_price
        self.alpha = alpha_loss
        if p0_mode == 'train':
            self.p0 = nn.Parameter(torch.tensor(1.96, requires_grad=True))
        else:
            self.p0 = 1.96
        self.p0_mode = p0_mode
        self.transaction_cost_rate = trans_cost_rate

    def terminal_payoff(self, final_price):
        return torch.max(final_price - self.K, torch.zeros_like(final_price))

    def forward(self, holding, S, F1, VIX,
                output_hedging_error=False, p0=None):
        """
        Args:
            holding: (N, T, 2) — [delta_S, delta_F1] per timestep
            S:  (N, T+1)
            F1: (N, T+1)
            p0: optional externally provided p0
        Returns:
            CVaR loss (scalar)
        """
        # Price changes
        delta_S = S[:, 1:] - S[:, :-1]      # (N, T)
        delta_F1 = F1[:, 1:] - F1[:, :-1]

        daily_roll_yield = (VIX[:, :-1] - F1[:, :-1]) * (1.0 / 30.0)

        true_delta_F1 = delta_F1 + daily_roll_yield

        # Stack: (N, T, 2)
        delta_price = torch.stack([delta_S, true_delta_F1], dim=-1)

        # PnL = sum over time and instruments
        PnL = (holding * delta_price).sum(dim=(1, 2))  # (N,)

        # Hedging cost
        C_T = self.terminal_payoff(S[:, -1])
        X = C_T - PnL

        # Transaction costs
        if self.transaction_cost_rate > 0:
            price_levels = torch.stack([S[:, :-1], F1[:, :-1]], dim=-1)
            delta_holding = torch.diff(holding, dim=1,
                                       prepend=holding[:, :1])
            tc = (delta_holding.abs() * self.transaction_cost_rate
                  * price_levels).sum(dim=(1, 2))
            X = X + tc

        # p0 determination
        if self.p0_mode == 'search':
            self.p0 = find_optimal_p0(X, self.alpha)
        elif self.p0_mode == 'given':
            if p0 is None:
                raise ValueError('p0 is not provided')
            self.p0 = p0
        elif self.p0_mode == 'train':
            pass

        X = X - self.p0
        loss = torch.max(X, torch.zeros_like(X)) / (1 - self.alpha) + self.p0

        if output_hedging_error:
            return loss.mean(), torch.max(X, torch.zeros_like(X)).mean()
        return loss.mean()


# ============================================================================
# 5. ADVERSARIAL ATTACKER — S-attack (perturb S only, VIX/F unchanged)
# ============================================================================

class VIX_Heston_Attacker():
    """
    Adversarial attacker for VIX hedging. Performs S-attack:
    perturbs stock price S only, keeps V/VIX/F1 unchanged.

    This is consistent with the existing Heston S_budget_attack approach.
    """
    def __init__(self, loss_fn, K, T, dt, sequence_length):
        self.loss_fn = loss_fn
        self.K = K
        self.T = T
        self.dt = dt
        self.sequence_length = sequence_length

    def performance(self, network, S, VIX, F1):
        """Evaluate hedging performance on given data."""
        features = construct_features(S, VIX, F1,
                                      self.K, self.T, self.dt, self.sequence_length)
        holding = network(features)
        loss = self.loss_fn(holding, S, F1)
        return loss.item()

    def S_budget_attack(self, network, S, VIX, F1, delta, ratio, n_iter):
        """
        Budget-based adversarial attack on S only.

        Finds worst-case perturbation of S within a Wasserstein ball of
        radius delta. VIX, F1 are kept unchanged (they depend on V
        which is not perturbed in S-attack).

        Args:
            network: hedging network
            S, VIX, F1: tensors of shape (N, T+1)
            delta: perturbation budget
            ratio: step size ratio
            n_iter: number of attack iterations

        Returns:
            S_att: perturbed stock prices
            VIX, F1: unchanged
        """
        if delta == 0:
            return S, VIX, F1

        # Initialize attack parameters
        budget = torch.ones(S.shape[0]).to(device) * delta
        budget.requires_grad = True
        att_sign = torch.ones_like(S).sign().to(device)
        att_sign_old = att_sign.clone().detach()
        att_sign[:, 0] = 0
        att_sign.requires_grad = True

        # Step size
        alpha = delta * ratio / n_iter

        # Best attack tracking
        att_best = (budget.unsqueeze(1) * att_sign).clone().detach()
        result_best = 0.

        for i in range(n_iter):
            S_att = S + budget.unsqueeze(1) * att_sign

            # Construct features with perturbed S
            features = construct_features(S_att, VIX, F1,
                                          self.K, self.T, self.dt, self.sequence_length)
            holding = network(features)
            loss = self.loss_fn(holding, S_att, F1)

            if loss.item() > result_best:
                att_best = (budget.unsqueeze(1) * att_sign).clone().detach()
                result_best = loss.item()

            grad_b = torch.autograd.grad(loss, budget, retain_graph=True)[0]
            grad_a = torch.autograd.grad(loss, att_sign, retain_graph=True)[0]

            with torch.no_grad():
                # Update budget
                budget_new = budget + alpha * grad_b.pow(2 - 1) * (
                    (grad_b.pow(2).mean() + 1e-10).pow(1 / 2 - 1))
                budget_new = budget_new / budget_new.square().mean().sqrt() * delta
                budget_new = torch.max(budget_new, torch.zeros_like(budget_new))
                budget.copy_(budget_new)

                # Update attack sign
                grad_a[:, 0] = 0
                att_sign.copy_(grad_a.sign())

        # Final evaluation
        S_att = S + budget.unsqueeze(1) * att_sign
        features = construct_features(S_att, VIX, F1,
                                      self.K, self.T, self.dt, self.sequence_length)
        holding = network(features)
        loss = self.loss_fn(holding, S_att, F1)
        if loss.item() > result_best:
            att_best = (budget.unsqueeze(1) * att_sign).clone().detach()

        S_att = S + att_best
        return S_att, VIX, F1


class loss_MSE_VIX(nn.Module):
    """
    Mean Squared Hedging Error for VIX Deep Hedging.
    Forces the agent to act strictly as a hedger, heavily penalizing 
    ANY deviation (profit or loss) from the option liability.
    """
    def __init__(self, Strike_price, trans_cost_rate=0.):
        super(loss_MSE_VIX, self).__init__()
        self.K = Strike_price
        self.tc_rate = trans_cost_rate
        # p0 represents the initial option premium collected
        self.p0 = nn.Parameter(torch.tensor(1.69))

    def terminal_payoff(self, final_price):
        return torch.max(final_price - self.K, torch.zeros_like(final_price))

    def forward(self, holding, S, F1, VIX):
        # 1. Price Changes & Structural Roll Yield
        delta_S = S[:, 1:] - S[:, :-1]
        delta_F1_constant = F1[:, 1:] - F1[:, :-1]
        
        daily_roll_yield = (VIX[:, :-1] - F1[:, :-1]) * (1.0 / 30.0)
        true_delta_F1 = delta_F1_constant + daily_roll_yield
        
        delta_price = torch.stack([delta_S, true_delta_F1], dim=-1)
        
        # 2. PnL & Liability
        PnL = (holding * delta_price).sum(dim=(1, 2))
        C_T = self.terminal_payoff(S[:, -1])
        
        # X represents the raw hedging shortfall (Liability - PnL)
        X = C_T - PnL
        
        # 3. Transaction Costs
        if self.tc_rate > 0:
            price_levels = torch.stack([S[:, :-1], F1[:, :-1] * 5.0], dim=-1)
            delta_holding = torch.diff(holding, dim=1, prepend=holding[:, :1])
            tc = (delta_holding.abs() * self.tc_rate * price_levels).sum(dim=(1, 2))
            X = X + tc  # Friction increases the shortfall
            
        # 4. Mean Squared Error
        # We want: Initial Premium (p0) + PnL - Costs = Liability (C_T)
        # Rearranged: p0 - X = 0
        # The agent minimizes the squared distance from perfect replication
        loss = torch.mean((X - self.p0) ** 2)
        
        return loss

