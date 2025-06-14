import pandas as pd
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

# Load example data (you can replace with actual paths)
ofi_A = pd.read_csv('ofi_asset_A.csv', index_col='timestamp')
returns_B = pd.read_csv('returns_asset_B.csv', index_col='timestamp')

# Join on timestamps
data = ofi_A.join(returns_B, how='inner')

# Extract features and target
X = data[['OFI']]
y = data['returns']

# Fit Lasso regression
model = Lasso(alpha=0.01)
model.fit(X, y)

# Print coefficient
print("Cross-Impact Coefficient (OFI_A → returns_B):", model.coef_[0])

# Optional: visualize the fit
plt.scatter(X, y, alpha=0.3)
plt.plot(X, model.predict(X), color='red')
plt.title("Cross-Asset OFI Impact")
plt.xlabel("OFI of Asset A")
plt.ylabel("Returns of Asset B")
plt.show()