import pandas as pd

# 1. Load the raw dataset
df = pd.read_csv('data/Crypto.csv')

# 2. Drop completely empty or unnamed columns
df = df.loc[:, ~df.columns.str.contains(r'^Unnamed')]

# 3. Convert 'marketcap' from string to numeric (remove any commas or currency symbols)
df['marketcap'] = (
    df['marketcap']
    .astype(str)
    .str.replace(r'[,\$]', '', regex=True)
    .replace('', pd.NA)
    .astype(float)
)

# 4. Convert 'date_taken' from string to datetime, assuming format 'dd-mm-yyyy'
df['date_taken'] = pd.to_datetime(df['date_taken'], format='%d-%m-%Y', errors='coerce')

# 5. Drop rows with more than 30% missing values
threshold = int(df.shape[1] * 0.7)  # require at least 70% non-null
df = df.dropna(thresh=threshold)

# 6. (Optional) Reset index after cleaning
df = df.reset_index(drop=True)

# 7. Save the cleaned dataset to a new CSV
cleaned_path = 'data/Crypto_cleaned.csv'
df.to_csv(cleaned_path, index=False)

# 8. Display basic info about the cleaned DataFrame
print("Cleaned DataFrame shape:", df.shape)
print(df.dtypes)

# Provide path to the cleaned file for downstream processing
print(f"Cleaned dataset saved to: {cleaned_path}")
