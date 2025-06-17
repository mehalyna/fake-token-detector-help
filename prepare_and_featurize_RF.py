import pandas as pd

# 1. Load cleaned datasets
df_valid = pd.read_csv('data/Crypto_cleaned.csv')
df_scam = pd.read_csv('data/urls_cleaned.csv')

# 2. Rename 'crypturl' to 'url' in the valid tokens dataset
df_valid = df_valid.rename(columns={'crypturl': 'url'})

# 3. Assign labels
df_valid['is_scam'] = 0
df_scam['is_scam'] = 1

# 4. Drop any extra columns to keep only one 'url' column in both
#    (valid dataset had no 'url' originally, scams dataset has 'url' already)
#    Ensure both dataframes have identical column sets
all_columns = set(df_valid.columns).union(df_scam.columns)
for col in all_columns:
    if col not in df_valid.columns:
        df_valid[col] = pd.NA
    if col not in df_scam.columns:
        df_scam[col] = pd.NA

# 5. Concatenate the dataframes
df_final = pd.concat([df_valid, df_scam], ignore_index=True, sort=False)

# 6. Drop any duplicate columns (if needed)
#    e.g., if there was an old 'crypturl' left over, ensure it's removed
if 'crypturl' in df_final.columns:
    df_final = df_final.drop(columns=['crypturl'])

# 7. Save the final labeled dataset with unified 'url' column
output_path = 'data/Crypto_final_labeled.csv'
df_final.to_csv(output_path, index=False)

# 8. Display summary
print("Final dataset shape:", df_final.shape)
print("Label distribution:")
print(df_final['is_scam'].value_counts())
print(f"Unified dataset saved to: {output_path}")
