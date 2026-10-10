import numpy as np

# Example healthcare-style data
ages = np.array([25, 40, 60], dtype=float)

claim_amounts = np.array([2000, 12000, 50000], dtype=float)

# Compute mean and std
age_mean = ages.mean()
age_std = ages.std()

claim_mean = claim_amounts.mean()
claim_std = claim_amounts.std()

# Z-score normalization
ages_z = (ages - age_mean) / age_std

claims_z = (claim_amounts - claim_mean) / claim_std

print("Original Ages:")
print(ages)

print("\nNormalized Ages:")
print(np.round(ages_z, 2))

print("\nOriginal Claim Amounts:")
print(claim_amounts)

print("\nNormalized Claim Amounts:")
print(np.round(claims_z, 2))