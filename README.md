# 2025 Fantasy Football QB Point Predictor
By: Ethan Ooi

The goal of this project is to predict QBs average fantasy points per game in the upcoming 2025 NFL Season.
This Fantasy Football Point Predictor uses NFL Data provided by https://www.pro-football-reference.com/. Fantasy data is collected from the 2004-2024 seasons. 

## Methodology
To predict average fantasy points per game for QBs in the 2025 NFL season, I created a machine learning pipeline using historical NFL data from the 2004-2024 seasons.

## Data Collection & Cleaning
I scraped data from https://www.pro-football-reference.com/ using [nfl_stats_scraper.py](nfl_stats_scraper.py). 
I cleaned this data by:
- handling nulls
- standardizing columns 
- removing extra noise from the dataset retaining only relevant data.

## Feature Engineering
I engineered relevent features for more effective model training:
- Per-game averages (Passing Yards/Game, etc.)
- Rolling averages (Past 2 & 3 Year averages)
- All Pro & Pro Bowl selections
- Rolling season number count by player

## Feature Selection
I used mlxtend's Sequential Feature Selector(SFS) to run a forward feature selection on all relevant quantitative variables. This helped me limit total training features to 15, while avoiding multicollinearity and overfitting in my model. 
The final features selected were:
`Selected Features: ['PassTDs', 'RushTDs', 'Fmb', '2PM', '2PP', 'PassTD/G', 'RushTD/G', 'AllPro', '#ofY', 'CompPropLast2Y', 'PassTD/GLast2Y', 'RushYds/GLast2Y', 'RushYds/GLast3Y', 'FantPts/GLast2Y', 'FantPts/GLast3Y']`

## Repository Guide
- [data](data): Holds data created from [nfl_stats_scraper.py](nfl_stats_scraper.py).
- [models](models): Holds the trained model .json file.
- [results](results): Holds the final 2025 prediction .csv and feature importance barplot.
- [scripts](scripts): Holds 4 scripts which contain functions used in [main.py](main.py).
- [main.py](main.py): Python script which runs the entire project.
- [nfl_stats_scraper.py](nfl_stats_scraper.py): Scrapes https://www.pro-football-reference.com/ for QB data, saves to [data](data).
- [README.md](README.md): This file.
- [requirements.txt](requirements.txt): Installation versions to properly run scripts.

## Requried Libraries
- Python 3.x
- pandas
- numpy
- xgboost==1.7.6
- scikit-learn
- mlxtend
- matploltlib
- seaborn

```bash
pip install pandas numpy xgboost==1.7.6 scikit-learn mlxtend matplotlib seaborn
