# build_enhanced_dataset.py

import pandas as pd
import numpy as np


df_valid = pd.read_csv('data/Crypto_cleaned.csv')
df_scam  = pd.read_csv('data/urls_cleaned.csv')


if 'crypturl' in df_valid.columns:
    df_valid = df_valid.rename(columns={'crypturl': 'url'})


def add_slim_features(df):

    df['name_length'] = df['name'].fillna('').str.len()

    df['digits_ratio'] = df['name'].fillna('').apply(
        lambda s: sum(c.isdigit() for c in s) / len(s) if len(s)>0 else 0
    )

    df['url_slug'] = df['url'].fillna('').apply(lambda u: u.rstrip('/').split('/')[-1])
    df['hyphens_count'] = df['url_slug'].str.count('-')
    return df

df_valid = add_slim_features(df_valid)
df_scam  = add_slim_features(df_scam)


numeric_cols = ['price','volume24hrs','marketcap','circulatingsupply','maxsupply','totalsupply']
df_num = df_valid[['url'] + numeric_cols].drop_duplicates(subset=['url'])
df_scam = df_scam.merge(df_num, on='url', how='left')


min_vals = df_num.drop(columns=['url']).min()
max_vals = df_num.drop(columns=['url']).max()
rng = np.random.default_rng(42)
for col in numeric_cols:
    mask = df_scam[col].isna()
    df_scam.loc[mask, col] = rng.uniform(min_vals[col], max_vals[col], size=mask.sum())


df_valid['is_scam'] = 0
df_scam ['is_scam'] = 1

all_cols = set(df_valid.columns) | set(df_scam.columns)
for c in all_cols:
    if c not in df_valid: df_valid[c] = pd.NA
    if c not in df_scam:  df_scam[c]  = pd.NA

df_final = pd.concat([df_valid, df_scam], ignore_index=True, sort=False)
df_final.to_csv('data/Crypto_enhanced_dataset.csv', index=False)
print("Saved Crypto_enhanced_dataset.csv with slim + numeric features.")
