import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
df = pd.DataFrame({
    'x': np.random.randn(100), 
    'y': np.random.randn(100)})

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Show regression line')
sns.lmplot(x='x', y='y', data=df, fit_reg=True, scatter_kws={"marker": "D", "s": 100})

plt.subplot(1, 2, 2)
sns.lmplot(x='x', y='y', data=df, fit_reg=False, scatter_kws={"marker": "D", "s": 100})
plt.title("Without line")
plt.show()