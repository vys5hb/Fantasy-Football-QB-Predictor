import pandas as pd

def cleaning_data(df):
    # Drop irrelevent columns & rename useful columns
    df.drop(columns=['Rk', 'Tm', 'Rec', 'Yds.2', 'Y/A', 'Y/R', 'TD.2', 'TD.3', 'FantPt', 'DKPt', 'FDPt', 'PosRank', 'VBD', 'Y/R', 'Tgt', 'OvRank'], inplace=True)
    df.rename(columns={
        'FantPos': 'Pos',
        'Att': 'PassAtts',
        'Yds': 'PassYds',
        'TD': 'PassTDs',
        'Int': 'Ints',
        'Att.1': 'RushAtts',
        'Yds.1': 'RushYds',
        'TD.1': 'RushTDs',
        'PPR': 'FantPts'
    }, inplace=True)

    # Convert quantitative variables to integers from strings
    convert_to_int = ['PassAtts', 'PassYds', 'PassTDs', 'Ints', 'RushAtts', 'RushYds', 'RushTDs', 'FantPts', 'G', 'GS', 'Age', 'Year', 'Cmp', 'Fmb', 'FL', '2PM', '2PP']
    for col in convert_to_int:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Sort for entries only from years where QBs have scored over 30 fantasy points since 2011
    df = df[df['Pos'] == 'QB'].copy()
    current_year = df['Year'].max()
    df = df[df['Year'] >= current_year - 13]
    df = df[df['FantPts'] >= 30].copy()
    
    # Create new features for season averages
    df.loc[:, 'CompProp'] = round(df['Cmp'] / df['PassAtts'], 2)
    df.loc[:, 'PassAtts/G'] = round(df['PassAtts'] / df['G'], 2)
    df.loc[:, 'PassYds/G'] = round(df['PassYds'] / df['G'], 2)
    df.loc[:, 'PassTD/G'] = round(df['PassTDs'] / df['G'], 2)
    df.loc[:, 'Turnovers/G'] = round((df['Ints'] + df['FL']) / df['G'], 2)
    df.loc[:, 'RushTD/G'] = round(df['RushTDs'] / df['G'], 2)
    df.loc[:, 'RushYds/G'] = round(df['RushYds'] / df['G'], 2)
    df.loc[:, 'FantPts/G'] = round(df['FantPts'] / df['G'], 2)
    
    # Use Pro Football Reference's "+" and "*" to assign "ProBowl" and "AllPro" players
    df['ProBowl'] = df['Player'].str.contains(r'\*').astype(int)
    df['AllPro'] = df['Player'].str.contains(r'\+').astype(int)
    df['Player'] = df['Player'].str.replace(r'[+*]', '', regex=True).str.strip()

    # Fill NaNs, adjust for divison by zeroes
    df.replace([float('inf'), -float('inf')], 0, inplace=True)
    df.fillna(0, inplace=True)
    df = df.reset_index(drop=True)

    return df

def season_count(df):
    # Create categorical variables for rookie player's seasons & total season number per player
    df['SeasonNumber'] = df.groupby('Player').cumcount() + 1
    df['#ofY'] = df['SeasonNumber'].apply(lambda x: 'rookie' if x ==1 else ('second' if x == 2 else '3 or more'))
    df['#ofY'] = df['#ofY'].astype('category').cat.codes

    return df

def rolling_features(df):
    # Create rolling averages for the previous 2 to 3 years
    rolling_features = ['CompProp', 'PassAtts/G', 'PassYds/G', 'PassTD/G', 'Turnovers/G', 'RushTD/G', 'RushYds/G', 'FantPts/G']
    for feature in rolling_features:
        df[f'{feature}Last2Y'] = round(df.groupby('Player')[feature].transform(lambda x: x.shift(1).rolling(2, min_periods=1).mean()), 3)
        df[f'{feature}Last3Y'] = round(df.groupby('Player')[feature].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()), 3)
        # Fill rookie & second-year player's "Last2Y" and "Last3Y" data with zeroes
        df.loc[df['#ofY'] == 1, [f'{feature}Last2Y', f'{feature}Last3Y']] = 0
        df.loc[df['#ofY'] == 2, f'{feature}Last3Y'] = 0
        
    return df

def next_season_pts(df):
    # Shift "FantPts/G" back a year to align variables when training model
    df['NextYearFantPt/G'] = df.groupby('Player')['FantPts/G'].shift(-1)
    df = df[df['NextYearFantPt/G'].notna()]
    
    return df