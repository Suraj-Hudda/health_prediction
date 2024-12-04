import pandas as pd
data=pd.read_csv('myData.csv')
df=pd.DataFrame(data)
transposed_df = df.T 
filename = 'transposed_health_data.csv'
transposed_df.to_csv(filename, header=False)  # Save without headers
print(f'\nTransposed data saved to {filename}')