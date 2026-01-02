import numpy as np
import pandas as pd
from scipy.stats import pearsonr, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

# Set print options for clarity
np.set_printoptions(suppress=True, precision = 2)

# Load NBA game data
nba = pd.read_csv('./nba_games.csv')

# Subset Data to 2010 Season, 2014 Season
nba_2010 = nba[nba.year_id == 2010]
nba_2014 = nba[nba.year_id == 2014]

# Display first few rows of each subset
print(nba_2010.head(), '\n')
print(nba_2014.head(), '\n')

# Extract points scored by Knicks and Nets in 2010 season
knicks_pts_10 = nba_2010.pts[nba_2010['fran_id'] == 'Knicks']
nets_pts_10 = nba_2010.pts[nba_2010['fran_id'] == 'Nets']

# Display points data
print('2010 Season Knicks Points:\n')
print(knicks_pts_10, '\n')
print('------------------\n')
print('2010 Season Nets Points:\n')
print(nets_pts_10, '\n')

# Calculate and print difference in means
diff_means_2010 = knicks_pts_10.mean() - nets_pts_10.mean()
print('Difference in Means (Knicks - Nets) for 2010 Season:' , diff_means_2010)

#diff_means_2010 = np.mean(knicks_pts_10) - np.mean(nets_pts_10) #alternative

# Plot histograms of points scored in 2010 season
plt.hist(knicks_pts_10, alpha=0.5, density=True, label='Knicks')
plt.hist(nets_pts_10, alpha=0.5, density=True, label='Nets')
plt.legend()
plt.title('2010 Season')
plt.show()
plt.close()

# Repeat analysis for 2014 season
knicks_pts_14 = nba_2014.pts[nba_2014['fran_id'] == 'Knicks']
nets_pts_14 = nba_2014.pts[nba_2014['fran_id'] == 'Nets']

diff_means_2014 = knicks_pts_14.mean() - nets_pts_14.mean()
print('Difference in Means (Knicks - Nets) for 2014 Season:' , diff_means_2014)

# Plot histograms of points scored in 2014 season
plt.hist(knicks_pts_14, alpha=0.5, density=True, label='Knicks')
plt.hist(nets_pts_14, alpha=0.5, density=True, label='Nets')
plt.legend()
plt.title('2014 Season')
plt.show()
plt.close()

# Boxplot of points scored by franchise in 2010 season
plt.clf()
sns.boxplot(data=nba_2010, y='pts', x='fran_id')
plt.show()

# Contingency table of game location vs game result in 2010 season
location_result_freq = pd.crosstab(nba_2010.game_location, nba_2010.game_result)
print('\nContingency Table of Game Location vs Game Result (2010 Season):\n')
print(location_result_freq)

# Proportions table
location_result_proportions = location_result_freq/len(location_result_freq)
print('\nProportions Table of Game Location vs Game Result (2010 Season):\n')
print(location_result_proportions)

# Chi-squared test of independence
chi2, pval, dof, expected = chi2_contingency(location_result_freq)
print('\nChi-squared Test Results (2010 Season):')
print('Chi2 Statistic:', chi2)
print('p-value:', pval)
print('Expected Frequencies:\n')
print(expected)

# Covariance and Correlation between forecasted win probability and point differential in 2010 season
point_diff_forecast_cov = np.cov(nba_2010.forecast, nba_2010.point_diff)
print('\nCovariance Matrix between Forecasted Win Prob. and Point Differential (2010 Season):\n')
print(point_diff_forecast_cov)

point_diff_forecast_corr, p = pearsonr(nba_2010.forecast, nba_2010.point_diff) 
print('\nPearson Correlation between Forecasted Win Prob. and Point Differential (2010 Season):\n')
print('Correlation Coefficient:', point_diff_forecast_corr)

# Scatter plot of forecasted win probability vs point differential in 2010 season
plt.scatter(nba_2010['forecast'], nba_2010['point_diff'])
plt.xlabel('Forecasted Win Prob.')
plt.ylabel('Point Differential')
plt.title('2010 Season: Forecasted Win Prob. vs Point Differential')
plt.show()
plt.clf()
