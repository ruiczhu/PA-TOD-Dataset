import pandas as pd
from datasets import load_dataset

ds = load_dataset("wenkai-li/big5_chat")

df = ds['train'].to_pandas()
df = df[['env_idx', 'trait', 'level', 'train_input', 'train_output']]

# 对 'trait' 和 'level' 列进行 One-Hot 编码
df_encoded = pd.get_dummies(df, columns=['trait', 'level'])

df_encoded.to_csv('processed_data_encoded.csv', index=False)
