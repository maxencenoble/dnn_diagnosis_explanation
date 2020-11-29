import numpy as np
import pandas as pd

series = {"maladie": ["A", "A","B","B"],
          "ecg": ["ecg1", "ecg2", "ecg1","ecg2"],
          "proba_base": [0.5, 0.5, 0.5, 0.5],
          "ecart_0.5": [0.25, 0.3, 0.3, 0],
          "ecart_moins0.5": [-0.25, 0.3, 0.5, 0]
          }
df1 = pd.DataFrame(series)

print(df1.set_index(["ecg", "maladie"]))

tau=np.array([-1.,-0.9])
a=np.array([1,2])
print(a[np.where(tau==-0.9)][0])
