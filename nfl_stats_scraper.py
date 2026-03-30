import pandas as pd
import time


def scrape_qb_stats(start_year=2004, end_year=2025, output_path='data/qb_data.csv'):
    all_years = []
    for year in range(start_year, end_year):
        url = f'https://www.pro-football-reference.com/years/{year}/fantasy.htm'
        print(f'Scraping {year}...')

        tables = pd.read_html(url, header=1)
        df = tables[0]
        df = df[df[df.columns[0]] != 'Player']
        df['Year'] = year
        df = df.reset_index(drop=True)
        all_years.append(df)
        time.sleep(1)

    qb_df = pd.concat(all_years, ignore_index=True)
    qb_df.to_csv(output_path, index=False)
    print(f'Saved {len(qb_df)} rows to {output_path}')


if __name__ == '__main__':
    scrape_qb_stats()
