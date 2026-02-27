# Micro-Mobility Demand Forecasting & Hypothesis Testing: Yulu

# Summary
This project applies advanced Hypothesis Testing (T-Tests, ANOVA, Chi-Square) to identify the statistical significance of factors affecting micro-mobility demand for Yulu, India's leading shared electric cycle provider. By analyzing weather patterns, seasonal fluctuations, and working-day behaviors, the analysis transitions from basic exploratory data to mathematically proven business drivers, enabling optimized fleet distribution and dynamic pricing strategies.

# Business Problem
Yulu recently experienced considerable dips in revenue and needed to understand the underlying factors driving electric cycle demand. The objective of this consulting project was to:
Identify which variables (Weather, Season, Working Day) are statistically significant in predicting customer demand.
Prove whether the perceived differences in daily rentals are mathematically real or just random variance.
Provide actionable recommendations to optimize fleet availability and mitigate revenue loss during low-demand periods.

#Technical Stack
Language: Python 3
Data Manipulation: Pandas, NumPy
Statistical Analysis: SciPy (scipy.stats for T-Test, ANOVA, Chi-Square, Shapiro-Wilk, Levene's Test)
Data Visualization: Seaborn, Matplotlib

# Part 1: 2-Sample T-Test (Working Day vs. Weekend Demand)
Business Question: Does a "Working Day" have a fundamentally different demand profile than a weekend or holiday?
Null Hypothesis (H0): There is NO statistical difference in the number of cycles rented on working days compared to non-working days.
Alternative Hypothesis (H1): There IS a statistical difference in the number of cycles rented.

import pandas as pd
from scipy.stats import ttest_ind

- Load data
df = pd.read_csv('yulu_data.csv')

-  Split data into Working Days and Non-Working Days
working_days = df[df['workingday'] == 1]['count']
non_working_days = df[df['workingday'] == 0]['count']

- Perform 2-Sample T-Test
t_stat, p_value = ttest_ind(working_days, non_working_days)

print(f"P-Value: {p_value}")
- Result: P-Value > 0.05 (Fail to reject Null Hypothesis)
- Insight: Surprisingly, the total aggregate demand does not significantly change whether it is a working day or a weekend. However, underlying EDA reveals the type of user changes (registered commuters on weekdays vs. casual users on weekends).

# Part 2: ANOVA (Analysis of Variance) for Multiple Categories

Business Question: Do different seasons or different weather conditions significantly impact the number of cycles rented?
(Because Season and Weather have more than 2 categories, we must use ANOVA instead of a T-Test).
Null Hypothesis (H0): Demand is equal across all weather conditions.
from scipy.stats import f_oneway

- Grouping demand by weather condition (1: Clear, 2: Mist, 3: Light Rain, 4: Heavy Rain/Snow)
weather_1 = df[df['weather'] == 1]['count']
weather_2 = df[df['weather'] == 2]['count']
weather_3 = df[df['weather'] == 3]['count']
weather_4 = df[df['weather'] == 4]['count']

- Perform ANOVA
f_stat, p_value = f_oneway(weather_1, weather_2, weather_3, weather_4)

print(f"P-Value: {p_value}")
# Result: P-Value < 0.05 (Reject Null Hypothesis)

- Insight: The P-value approaches 0, proving definitively that weather significantly impacts demand. Clear weather yields the highest rentals, while Light/Heavy Rain drops demand drastically.

# Part 3: Chi-Square Test for Independence

Business Question: Are Weather and Season dependent on each other? This helps prevent multicollinearity in future predictive modeling.
from scipy.stats import chi2_contingency

- Create a contingency table between Season and Weather
contingency_table = pd.crosstab(df['season'], df['weather'])

- Perform Chi-Square Test
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

print(f"P-Value: {p_value}")
- Result: P-Value < 0.05 (Reject Null Hypothesis)

- Insight: Weather and Season are statistically dependent. This means any future Machine Learning models (like Linear Regression) must account for this dependency to avoid skewed predictions.

# Strategic Recommendations

- Dynamic Fleet Rebalancing: Since aggregate volume doesn't change on weekends, but the user type does, Yulu should rebalance fleets away from corporate tech parks on Friday nights and move them towards recreational zones and parks for casual weekend riders.
- Weather-Triggered Promotions: ANOVA proved extreme drop-offs during adverse weather. Implement a "Surge Discount" (e.g., 20% off) automatically triggered in the app during Category 2 (Mist/Cloudy) weather to incentivize riders who are on the fence about renting.
- Inventory Maintenance Scheduling: Category 3 and 4 weather (Heavy Rain/Snow) have near-zero statistical demand. Yulu should use advanced weather forecasting to systematically pull cycles off the street for battery replacement and physical maintenance during these specific weather windows to minimize operational downtime.
