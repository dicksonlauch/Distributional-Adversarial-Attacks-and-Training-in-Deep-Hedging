import torch
from VIX_Heston_util import VIXHestonPathGenerator

# Heston model parameters (same as existing Heston pipeline)
sequence_length = 30
dt = 1/365
T = dt * sequence_length
K = 100
s0 = 100
v0 = 0.04
alpha = 1.
b = 0.04
sigma = 2
rho = -0.7

# VIX parameters — fixed 30-day tenor
tau_vix = 30/365
tau_fut1 = 30/365   # front-month VIX future

# Create path generator (precomputes VIX futures lookup tables via quadrature)
generator = VIXHestonPathGenerator(
    s0=s0, v0=v0, alpha=alpha, b=b, sigma=sigma, rho=rho,
    timestep=sequence_length, T=T,
    tau_vix=tau_vix, tau_fut1=tau_fut1
)

# Generate datasets
print("Generating training data (100K paths)...")
data_train = generator.generate(100000)
print(f"  S: {data_train[0].shape}, V: {data_train[1].shape}, "
      f"VIX: {data_train[2].shape}, F1: {data_train[3].shape}")
torch.save(data_train, '../Data/VIX_Heston_train.pt')

print("Generating test data (1M paths)...")
data_test = generator.generate(1000000)
print(f"  S: {data_test[0].shape}, V: {data_test[1].shape}, "
      f"VIX: {data_test[2].shape}, F1: {data_test[3].shape}")
torch.save(data_test, '../Data/VIX_Heston_test.pt')

print("Generating validation data (100K paths)...")
data_val = generator.generate(100000)
print(f"  S: {data_val[0].shape}, V: {data_val[1].shape}, "
      f"VIX: {data_val[2].shape}, F1: {data_val[3].shape}")
torch.save(data_val, '../Data/VIX_Heston_val.pt')

print("All datasets saved to Data/")
