import pandas as pd

df = pd.read_csv("raw_data/property.csv")  

df['price'] = pd.to_numeric(df['price'], errors='coerce')
df = df.dropna(subset=['price', 'proType', 'bedroom', 'bathroom', 'parking'])

for col in ['bedroom', 'bathroom', 'parking']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna()

price_threshold = df['price'].quantile(0.99)
df_filtered = df[df['price'] <= price_threshold]

df_encoded = pd.get_dummies(df_filtered, columns=['proType'], drop_first=True)

df_encoded.to_csv("raw_data/cleaned_encoded_property_data.csv", index=False)
