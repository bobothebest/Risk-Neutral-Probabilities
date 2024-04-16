
#%% import repertories
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import pandas_datareader.data as web
from datetime import datetime, timedelta
import scipy.stats as stats
from sklearn.metrics import brier_score_loss, roc_curve, auc, log_loss
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures

#%%1. DataCleaning
# read "mpd_stats.csv" with relative path
mpd_stats = pd.read_csv("Input_Data/mpd_stats.csv", skiprows=2, header = 1)
# convert the "Date" column to datetime
mpd_stats["idt"] = pd.to_datetime(mpd_stats["idt"])
#rename the idt column to Date
mpd_stats.rename(columns = {"idt":"Date"}, inplace = True)

mpd_stats.info()

# check for missing values
mpd_stats.isnull().sum()

# # Filter rows where 'Date' is between 2010 and 2023
# Filtering the dataset to include only the records from the year 2012 onwards
filtered_mpd_stats = mpd_stats[(mpd_stats['Date'].dt.year >= 2012)].copy()
filtered_mpd_stats.isnull().sum()

# Extract year from Date for easier analysis
filtered_mpd_stats['Year'] = filtered_mpd_stats['Date'].dt.year

# Identify markets that have data for each year in the range 2013 to 2023
years = range(2012, 2024)  # 2023 is inclusive
markets_with_complete_data = filtered_mpd_stats.groupby('market').filter(lambda x: all(y in x['Year'].values for y in years))['market'].unique()

# Filter the original dataframe to keep only the markets identified
mpd_stats_filtered_complete = mpd_stats[mpd_stats['market'].isin(markets_with_complete_data)]
mpd_stats_filtered_complete.isnull().sum()

# # remove rows that date is before 2013 jan 10
mpd_stats_filtered_complete = mpd_stats_filtered_complete[mpd_stats_filtered_complete['Date'] >= '2013-01-10']

print("Number of Unique market variables: ", len(mpd_stats_filtered_complete['market'].unique()))
print("Listed Market Variables: ", mpd_stats_filtered_complete['market'].unique())

# print the staring and ending date for each market in mpd_stats_filtered_complete
start_end_dates = mpd_stats_filtered_complete.groupby('market').agg(start_date=('Date', 'min'), end_date=('Date', 'max'))
print(start_end_dates)


#Upsampling for weekly data by forward filling, as majority of the market data frequency are in weekly basis
mpd_stats_filtered_complete.set_index('Date', inplace=True)

# fill data
def resample_and_fill(group):
    # Define the cutoff date
    cutoff_date = '2014-09-04'
    
    # Filter the group for dates before the cutoff
    group_before_cutoff = group[:cutoff_date]
    
    # Resample to weekly frequency and forward fill
    group_resampled = group_before_cutoff.resample('W-Thu').ffill()
    
    # Concatenate with the part of the group after the cutoff date
    group_after_cutoff = group[pd.to_datetime(cutoff_date) + pd.Timedelta(days=1):]
    result = pd.concat([group_resampled, group_after_cutoff])
    
    return result

# Apply the function to each group and combine the results
mpd_stats_weekly = mpd_stats_filtered_complete.groupby('market', group_keys=False).apply(resample_and_fill)

# Reset the index
mpd_stats_weekly.reset_index(inplace=True)

mpd_stats_weekly.isnull().sum()
#把maturity_target用前向填充填充掉
mpd_stats_weekly.fillna(method='ffill', inplace=True)
mpd_stats_weekly.isnull().sum()



#%%2.Feature Engineering
mpd_stats_weekly['market'].unique()
mpd_stats_weekly.info()

# create a df for each market with column name of market
features = pd.DataFrame(mpd_stats_weekly['market'].unique(), columns=['market'])
# add a column for with the name of feature_id
features['feature_name'] = mpd_stats_weekly['market'].unique()
# rename the market column to market category
features.rename(columns = {"market":"market_category"}, inplace = True)

# if the feature_id has "bac" 'citi', then change market_category to "Bank"
features.loc[features['feature_name'].isin(['bac', 'citi']), 'market_category'] = 'Bank'
# if the feature_id has "silver", "corn", "soybns", "gold", 'iyr', 'wheat', then change market_category to "Commodity"
features.loc[features['feature_name'].isin(['silver', 'corn', 'soybns', 'gold', 'iyr', 'wheat']), 'market_category'] = 'Commodity'
# if the feature_id has 'yen', 'euro', 'pound', then change market_category to "Currency"
features.loc[features['feature_name'].isin(['yen', 'euro', 'pound']), 'market_category'] = 'Currency'
# if the feature_id has 'sp12m', 'sp6m', then change market_category to "Equity"
features.loc[features['feature_name'].isin(['sp12m', 'sp6m']), 'market_category'] = 'Equity'
# if the feature_id has 'infl1y',  'infl2y', 'infl5y', then change market_category to "Inflation"
features.loc[features['feature_name'].isin(['infl1y', 'infl2y', 'infl5y']), 'market_category'] = 'Inflation'
# if the feature_id has 'tr10yr', 'tr5yr', 'LR3y3m','LR5y3m' then change market_category to "Rates"
features.loc[features['feature_name'].isin(['tr10yr', 'tr5yr', 'LR3y3m','LR5y3m']), 'market_category'] = 'Rates'

# group the features by market_category
features = features.sort_values(by='feature_name')
# add id number for each feature
features['id'] = range(1, len(features) + 1)
features

# Create a dictionary to map market names to feature IDs
market_to_id = features.set_index('feature_name')['id'].to_dict()

# Initialize df_feature_engineered with the Date column
df_feature_engineered = pd.DataFrame(mpd_stats_weekly[mpd_stats_weekly['market'] == 'bac']['Date'].copy()).drop_duplicates()

# Iterate over each market to merge its data into df_feature_engineered
for market in mpd_stats_weekly['market'].unique():
    # Select the data for the current market, excluding the 'market' column
    market_data = mpd_stats_weekly[mpd_stats_weekly['market'] == market].drop(columns=['market']).copy()
    
    # Get the feature ID for the current market
    feature_id = market_to_id[market]
    
    # Rename the columns based on the feature ID, except for 'Date'
    market_data.rename(columns={col: f'f{feature_id}_{col}' for col in market_data.columns if col != 'Date'}, inplace=True)
    
    # Merge the data into df_feature_engineered
    df_feature_engineered = df_feature_engineered.merge(market_data, on='Date', how='left')

# This ensures that the 'market' column is not included in the final DataFrame

# find nan values or missing values
nan_rows = df_feature_engineered.isnull().sum()

# Interpolate missing values linearly without using future data
df_feature_engineered['Date'] = pd.to_datetime(df_feature_engineered['Date'])
df_feature_engineered.set_index('Date', inplace=True)
df_feature_engineered.interpolate(method='time', inplace=True)

##################################################################################################################################################################

# solve quesiton 1 if the estimators are useful
sp500_data_path = 'Input_Data/^SPX.csv'
sp500_data = pd.read_csv(sp500_data_path)

# Convert the 'Date' column to datetime format
sp500_data['Date'] = pd.to_datetime(sp500_data['Date'])

# Filter the data for the specified date range
start_date = '2013-01-10'
end_date = '2024-02-07'
sp500_filtered_data = sp500_data[(sp500_data['Date'] >= start_date) & (sp500_data['Date'] <= end_date)]

# Calculate the 6-month return
# First, ensure the data is sorted by date
sp500_filtered_data = sp500_filtered_data.sort_values('Date')

# Calculate the 6-month return. The shift will be based on trading days (~126 trading days in 6 months)
sp500_filtered_data['6m_return'] = sp500_filtered_data['Adj Close'].pct_change(periods=126).shift(-126)

sp500_return = sp500_filtered_data[['Date', '6m_return']].dropna()

sp500_return['log_6m_return'] = np.log(1+sp500_return['6m_return'])

mpd_stats_weekly.info
mpd_stats_weekly.isnull().sum()

# Filter the DataFrame for market 'sp6m'
df_sp6m = mpd_stats_weekly[mpd_stats_weekly['market'] == 'sp6m']

# Filter the DataFrame for market 'sp12m'
df_sp12m = mpd_stats_weekly[mpd_stats_weekly['market'] == 'sp12m']

## (3) compared the real return and the predicted mean

# Merge to find matching dates and get the corresponding returns
merged_data_sp6m = pd.merge(df_sp6m, sp500_return, on='Date', how='left')

# Merge to find matching dates and get the corresponding returns
merged_data_sp12m = pd.merge(df_sp12m, sp500_return, on='Date', how='left')

# Check if the 'Date' columns in both dataframes are exactly the same
dates_match = merged_data_sp6m['Date'].equals(merged_data_sp12m['Date'])
dates_match

# Renaming 'mu' columns to 'mu_sp6m' and 'mu_sp12m' respectively and merging.
sp_y = pd.merge(
    merged_data_sp6m[['Date', 'p50', 'p10', 'p90', '6m_return']],
    merged_data_sp12m[['Date', 'p50', 'p10', 'p90']],
    on='Date',
    suffixes=('_sp6m', '_sp12m')
)

# Drop any rows with NaN values to clean up the data.
sp_y.dropna(inplace=True)

# Plot all the selected columns in the sp_y dataframe.
plt.figure(figsize=(14, 7))

# Plotting the median (p50) as solid lines.
plt.plot(sp_y['Date'], sp_y['p50_sp6m'], label='Median 6m', color='Teal', linewidth=2)
plt.plot(sp_y['Date'], sp_y['p50_sp12m'], label='Median 12m', color='DarkOrange', linewidth=2)

# Plotting the actual return (6m_return) as a solid line.
plt.plot(sp_y['Date'], sp_y['6m_return'], label='Actual 6m Return', color='CornflowerBlue', linewidth=2)

# Plotting the 10th and 90th percentiles (p10, p90) as dashed lines.
plt.plot(sp_y['Date'], sp_y['p10_sp6m'], label='10th Percentile 6m', color='slategray', linestyle='--')
plt.plot(sp_y['Date'], sp_y['p90_sp6m'], label='90th Percentile 6m', color='slategray', linestyle='--')

# Filling the area between the 10th and 90th percentiles with a shaded region.
plt.fill_between(sp_y['Date'], sp_y['p10_sp6m'], sp_y['p90_sp6m'], color='slategray', alpha=0.1)

# Optionally, do the same for the 12-month predictions, if needed.
# plt.plot(sp_y['Date'], sp_y['p10_sp12m'], label='10th Percentile 12m', color='peachpuff', linestyle='--')
# plt.plot(sp_y['Date'], sp_y['p90_sp12m'], label='90th Percentile 12m', color='peachpuff', linestyle='--')
# plt.fill_between(sp_y['Date'], sp_y['p10_sp12m'], sp_y['p90_sp12m'], color='peachpuff', alpha=0.3)

# Setting the title and labels.
plt.title('S&P 500 Median Predicted vs Actual Returns Over Time')
plt.xlabel('Date')
plt.ylabel('Values')

# Adding the legend outside the plot to not obscure data.
# Adding the legend outside the plot to not obscure data.
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()



## (4) compare the difference through the confidence interval
# create a function to calculate the accuracy of the predictions
def calculate_prediction_accuracy(df, actual_col, p10_col, p50_col, p90_col):
    # Count how many times the actual log return falls within the p10 and p90 interval
    within_interval = df.apply(lambda row: row[p10_col] <= row[actual_col] <= row[p90_col], axis=1).sum()
    # Count how many times the actual log return is greater than the median prediction
    above_median = (df[actual_col] > df[p50_col]).sum()
    # Calculate the total number of points
    total_points = df.shape[0]
    
    # Calculate the accuracy percentages
    interval_accuracy = within_interval / total_points
    median_accuracy = above_median / total_points
    
    return {
        'within_interval_accuracy': interval_accuracy,
        'above_median_accuracy': median_accuracy,
        'total_points': total_points
    }

# Apply the function to our merged data
# Assuming the actual log returns column in merged_data is 'log_6m_return'
accuracy_results_sp6m = calculate_prediction_accuracy(
    merged_data_sp6m, 
    actual_col='log_6m_return', 
    p10_col='p10', 
    p50_col='p50', 
    p90_col='p90'
)

print('the accuracy of sp6m predictions',accuracy_results_sp6m)
'''
{'within_interval_accuracy': 0.8615916955017301,
 'above_median_accuracy': 0.6591695501730104,
 'total_points': 578}
'''
accuracy_results_sp12m = calculate_prediction_accuracy(
    merged_data_sp12m, 
    actual_col='log_6m_return', 
    p10_col='p10', 
    p50_col='p50', 
    p90_col='p90'
)

print('the accuracy of sp12m predictions',accuracy_results_sp12m)
accuracy_results_sp12m

'''
{'within_interval_accuracy': 0.9446366782006921,
 'above_median_accuracy': 0.6003460207612457,
 'total_points': 578}
'''

## (5) check the accuracy of the direction of the prediction
# Create a new column to store the sign of the 6-month log return
# Convert the 'log_6m_return' column to +1 if positive, -1 if negative
sp500_return['log_6m_return_sign'] = sp500_return['log_6m_return'].apply(lambda x: 1 if x > 0 else -1)
sp500_return

# Calculate the probability of a positive return
# Function to calculate z-score adjusted for skewness and kurtosis using Cornish-Fisher expansion
def z_score_cf_expansion(q, skew, kurt):
    # z is the z-score for the standard normal distribution
    z = stats.norm.ppf(q)
    
    # Cornish-Fisher expansion
    z_adj = z + (z**2 - 1) * skew / 6 + (z**3 - 3 * z) * (kurt - 3) / 24 - (2 * z**3 - 5 * z) * (skew**2) / 36
    return z_adj

# Calculate the z-score for the 50th percentile (median)
q_median = 0.5
merged_data_sp6m['z_median'] = z_score_cf_expansion(q_median, merged_data_sp6m['skew'], merged_data_sp6m['kurt'])

# Calculate the adjusted probability of a positive return using the adjusted z-score for the 50th percentile
merged_data_sp6m['adj_prob_positive_return'] = 1 - stats.norm.cdf(merged_data_sp6m['z_median'])

# Display the new DataFrame with the calculated adjusted probabilities
merged_data_sp6m[['Date', 'mu', 'sd', 'skew', 'kurt', 'adj_prob_positive_return']]

###
merged_data_sp12m['z_median'] = z_score_cf_expansion(q_median, merged_data_sp12m['skew'], merged_data_sp12m['kurt'])

# Calculate the adjusted probability of a positive return using the adjusted z-score for the 50th percentile
merged_data_sp12m['adj_prob_positive_return'] = 1 - stats.norm.cdf(merged_data_sp12m['z_median'])

# Display the new DataFrame with the calculated adjusted probabilities
merged_data_sp12m[['Date', 'mu', 'sd', 'skew', 'kurt', 'adj_prob_positive_return']]


# Calculate the Brier score, ROC AUC, and Log Loss for the 6-month predictions

# Convert actual sign of return to binary (1 for positive, 0 for non-positive)
sp500_return['actual_binary_outcome'] = sp500_return['log_6m_return_sign'].apply(lambda x: 1 if x == 1 else 0)

##sp6m
# Merge the actual outcomes with the predicted probabilities
eval_data_sp6m = pd.merge(merged_data_sp6m, sp500_return, on='Date')

# Calculate Brier score
brier_score_sp6m = brier_score_loss(eval_data_sp6m['actual_binary_outcome'], eval_data_sp6m['adj_prob_positive_return'])

# Calculate ROC curve and AUC
fpr_sp6m, tpr_sp6m, thresholds_sp6m = roc_curve(eval_data_sp6m['actual_binary_outcome'], eval_data_sp6m['adj_prob_positive_return'])
roc_auc_sp6m = auc(fpr_sp6m, tpr_sp6m)

# Calculate Log Loss
logloss_sp6m = log_loss(eval_data_sp6m['actual_binary_outcome'], eval_data_sp6m['adj_prob_positive_return'])

# Print the evaluation metrics
(brier_score_sp6m, roc_auc_sp6m, logloss_sp6m)
#(0.30298877039810873, 0.6980581576893052, 0.8002290079689729)

'''

1.  **Brier Score**: This score ranges from 0 to 1, where 0 represents a perfect model and 1 represents the worst model.  A Brier score of 0.3030 indicates that, on average, the squared difference between the predicted probabilities and the actual outcomes is moderately high.  This suggests that the probabilities are not as close to the true outcomes as they could be, indicating room for improvement in the model.

2.  **ROC AUC**: The area under the ROC curve (AUC) is a measure of the model's ability to correctly classify the binary outcomes.  It ranges from 0.5 (no better than random chance) to 1 (perfect classification).  An AUC of 0.6981 indicates that the model has a moderate ability to discriminate between positive and non-positive returns.  This is better than random chance but not close to perfect.

3.  **Log Loss**: Also known as logistic loss or cross-entropy loss, this metric measures the performance of a classification model where the prediction is a probability between 0 and 1.  The log loss increases as the predicted probability diverges from the actual label.  A log loss of 0.8002 is relatively high, suggesting that the model's predicted probabilities could be more accurate.

In summary, while the model shows some predictive ability (as indicated by an AUC greater than 0.5), it is not highly accurate in predicting the sign of the returns, and there is a considerable probability error (as indicated by the Brier score and Log Loss).  The model may benefit from calibration to improve probability estimates or from incorporating additional features or different modeling techniques to enhance its predictive power.
'''

##sp12m
# Merge the actual outcomes with the predicted probabilities
eval_data_sp12m = pd.merge(merged_data_sp12m, sp500_return, on='Date')

# Calculate Brier score
brier_score_sp12m = brier_score_loss(eval_data_sp12m['actual_binary_outcome'], eval_data_sp12m['adj_prob_positive_return'])

# Calculate ROC curve and AUC
fpr_sp12m, tpr_sp12m, thresholds_sp12m = roc_curve(eval_data_sp12m['actual_binary_outcome'], eval_data_sp12m['adj_prob_positive_return'])
roc_auc_sp12m = auc(fpr_sp12m, tpr_sp12m)

# Calculate Log Loss
logloss_sp12m = log_loss(eval_data_sp12m['actual_binary_outcome'], eval_data_sp12m['adj_prob_positive_return'])

# Print the evaluation metrics
(brier_score_sp12m, roc_auc_sp12m, logloss_sp12m)
# (0.30458583771273107, 0.716120218579235, 0.8034942614886871)

## (6) Assessing the accuracy of significant rise and fall predictions
# The issue of significant rises and falls is similar to that of positive and negative returns.
# The first step is to calculate how many instances of significant increases or decreases actually occurred, marking them as 1 if they did, and 0 if they did not.
# Define the threshold for significant change as 20%
large_change_threshold = 0.20

# calculate large increase and large decrease
sp500_return['large_increase'] = sp500_return['log_6m_return'].apply(lambda x: 1 if x >= large_change_threshold else 0)
sp500_return['large_decrease'] = sp500_return['log_6m_return'].apply(lambda x: 1 if x <= -large_change_threshold else 0)

# print result
print(sp500_return.head())
sp500_return.describe()

# Then I already have the prediction for the rise and fall, and I will compare it with this prediction of rise and fall. The accuracy is also judged using brier_score, roc_auc, logloss.
# I should have four sets of data: 6m rise, 6m fall; 12m rise, 12m fall
#6m increase 
eval_data_sp6m = pd.merge(merged_data_sp6m, sp500_return, on='Date')
# Calculate Brier score
brier_score_sp6m_in = brier_score_loss(eval_data_sp6m['large_increase'], eval_data_sp6m['prInc'])

# Calculate ROC curve and AUC
fpr_sp6m_in, tpr_sp6m_in, thresholds_sp6m_in = roc_curve(eval_data_sp6m['large_increase'], eval_data_sp6m['prInc'])
roc_auc_sp6m_in = auc(fpr_sp6m_in, tpr_sp6m_in)

# Calculate Log Loss
logloss_sp6m_in = log_loss(eval_data_sp6m['large_increase'], eval_data_sp6m['prInc'])

# Print the evaluation metrics
(brier_score_sp6m_in, roc_auc_sp6m_in, logloss_sp6m_in)
#(0.012429016340377705, 0.9888475836431226, 0.05116827792489753)

#6m decrease
# Calculate Brier score
brier_score_sp6m_de = brier_score_loss(eval_data_sp6m['large_decrease'], eval_data_sp6m['prDec'])

# Calculate ROC curve and AUC
fpr_sp6m_de, tpr_sp6m_de, thresholds_sp6m_de = roc_curve(eval_data_sp6m['large_decrease'], eval_data_sp6m['prDec'])
roc_auc_sp6m_de = auc(fpr_sp6m_de, tpr_sp6m_de)

# Calculate Log Loss
logloss_sp6m_de = log_loss(eval_data_sp6m['large_decrease'], eval_data_sp6m['prDec'])

# Print the evaluation metrics
(brier_score_sp6m_de, roc_auc_sp6m_de, logloss_sp6m_de)
#(0.013842216166729912, 0.6867158671586716, 0.09762668168530149)

#12m increase
eval_data_sp12m = pd.merge(merged_data_sp12m, sp500_return, on='Date')
# Calculate Brier score
brier_score_sp12m_in = brier_score_loss(eval_data_sp12m['large_increase'], eval_data_sp12m['prInc'])

# Calculate ROC curve and AUC
fpr_sp12m_in, tpr_sp12m_in, thresholds_sp12m_in = roc_curve(eval_data_sp12m['large_increase'], eval_data_sp12m['prInc'])
roc_auc_sp12m_in = auc(fpr_sp12m_in, tpr_sp12m_in)

# Calculate Log Loss
logloss_sp12m_in = log_loss(eval_data_sp12m['large_increase'], eval_data_sp12m['prInc'])

# Print the evaluation metrics
(brier_score_sp12m_in, roc_auc_sp12m_in, logloss_sp12m_in)
#(0.019231624611220105, 0.9560099132589839, 0.11040701467531214)

#12m decrease
# Calculate Brier score
brier_score_sp12m_de = brier_score_loss(eval_data_sp12m['large_decrease'], eval_data_sp12m['prDec'])

# Calculate ROC curve and AUC
fpr_sp12m_de, tpr_sp12m_de, thresholds_sp12m_de = roc_curve(eval_data_sp12m['large_decrease'], eval_data_sp12m['prDec'])
roc_auc_sp12m_de = auc(fpr_sp12m_de, tpr_sp12m_de)

# Calculate Log Loss
logloss_sp12m_de = log_loss(eval_data_sp12m['large_decrease'], eval_data_sp12m['prDec'])

# Print the evaluation metrics
(brier_score_sp12m_de, roc_auc_sp12m_de, logloss_sp12m_de)
#(0.023975969887039705, 0.6841328413284133, 0.1547707960291045)
# The predictions for increase are accurate, but not as accurate for decrease.

# Calculate ROC curves and AUCs for all cases
fpr_sp6m_in, tpr_sp6m_in, _ = roc_curve(eval_data_sp6m['large_increase'], eval_data_sp6m['prInc'])
roc_auc_sp6m_in = auc(fpr_sp6m_in, tpr_sp6m_in)

fpr_sp6m_de, tpr_sp6m_de, _ = roc_curve(eval_data_sp6m['large_decrease'], eval_data_sp6m['prDec'])
roc_auc_sp6m_de = auc(fpr_sp6m_de, tpr_sp6m_de)

fpr_sp12m_in, tpr_sp12m_in, _ = roc_curve(eval_data_sp12m['large_increase'], eval_data_sp12m['prInc'])
roc_auc_sp12m_in = auc(fpr_sp12m_in, tpr_sp12m_in)

fpr_sp12m_de, tpr_sp12m_de, _ = roc_curve(eval_data_sp12m['large_decrease'], eval_data_sp12m['prDec'])
roc_auc_sp12m_de = auc(fpr_sp12m_de, tpr_sp12m_de)

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# 6m increase
axs[0, 0].plot(fpr_sp6m_in, tpr_sp6m_in, color='blue', label=f'ROC curve (area = {roc_auc_sp6m_in:.2f})')
axs[0, 0].plot([0, 1], [0, 1], color='navy', linestyle='--')
axs[0, 0].set_xlim([0.0, 1.0])
axs[0, 0].set_ylim([0.0, 1.05])
axs[0, 0].set_xlabel('False Positive Rate')
axs[0, 0].set_ylabel('True Positive Rate')
axs[0, 0].set_title('6m Increase ROC')
axs[0, 0].legend(loc="lower right")

# 6m decrease
axs[0, 1].plot(fpr_sp6m_de, tpr_sp6m_de, color='blue', label=f'ROC curve (area = {roc_auc_sp6m_de:.2f})')
axs[0, 1].plot([0, 1], [0, 1], color='navy', linestyle='--')
axs[0, 1].set_xlim([0.0, 1.0])
axs[0, 1].set_ylim([0.0, 1.05])
axs[0, 1].set_xlabel('False Positive Rate')
axs[0, 1].set_ylabel('True Positive Rate')
axs[0, 1].set_title('6m Decrease ROC')
axs[0, 1].legend(loc="lower right")

# 12m increase
axs[1, 0].plot(fpr_sp12m_in, tpr_sp12m_in, color='green', label=f'ROC curve (area = {roc_auc_sp12m_in:.2f})')
axs[1, 0].plot([0, 1], [0, 1], color='darkgreen', linestyle='--')
axs[1, 0].set_xlim([0.0, 1.0])
axs[1, 0].set_ylim([0.0, 1.05])
axs[1, 0].set_xlabel('False Positive Rate')
axs[1, 0].set_ylabel('True Positive Rate')
axs[1, 0].set_title('12m Increase ROC')
axs[1, 0].legend(loc="lower right")

#12m decrease
axs[1, 1].plot(fpr_sp12m_de, tpr_sp12m_de, color='green', label=f'ROC curve (area = {roc_auc_sp12m_de:.2f})')
axs[1, 1].plot([0, 1], [0, 1], color='darkgreen', linestyle='--')
axs[1, 1].set_xlim([0.0, 1.0])
axs[1, 1].set_ylim([0.0, 1.05])
axs[1, 1].set_xlabel('False Positive Rate')
axs[1, 1].set_ylabel('True Positive Rate')
axs[1, 1].set_title('12m Decrease ROC')
axs[1, 1].legend(loc="lower right")

#Adjust layout
plt.tight_layout()
plt.show()
