import pandas as pd
import time

all_years = []
for year in range(2004, 2025):
    url = f'https://www.pro-football-reference.com/years/{year}/fantasy.htm'
    print(f'{year}: ')
    
    tables = pd.read_html(url, header=1)
    df = tables[0]
    df = df[df[df.columns[0]] != 'Player']
    df['Year'] = year
    df = df.reset_index(drop=True)
    all_years.append(df)
    time.sleep(1)

qb_df = pd.concat(all_years, ignore_index=True)
qb_df.to_csv('data/qb_data.csv', index=False)