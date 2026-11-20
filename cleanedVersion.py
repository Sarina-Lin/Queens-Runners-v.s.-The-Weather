# ============================================================
# QUEENS RUNNERS VS. THE WEATHER
# Cross-Country 5K Performance Analysis
# Seasons: 2023-2024, 2024-2025, 2025-2026
# ============================================================

import pandas as pd
from scipy.stats import linregress
import statsmodels.formula.api as smf
import sys
import pandas
import numpy
import scipy
import statsmodels
import patsy

# ============================================================
# STEP 1: LOAD THE EXCEL WORKBOOK
# ============================================================

file_path = "Girls and Boys Cross Country PSAL Data.xlsx"

all_sheets = pd.read_excel(file_path, sheet_name=None)

print("WORKSHEETS FOUND:")
print(list(all_sheets.keys()))


# ============================================================
# STEP 2: COMBINE ALL SIX WORKSHEETS
# ============================================================

combined_data = []

for sheet_name, data in all_sheets.items():

    data = data.copy()

    # Remove extra spaces from column names
    data.columns = data.columns.str.strip()

    # Standardize inconsistent column names
    data = data.rename(columns={
        "Time": "Time (min:sec)",
        "Date": "Date (4:00 P.M.)"
    })

    # Identify gender and season from worksheet name
    gender, season = sheet_name.split(" ", 1)

    data["Gender"] = gender
    data["Season"] = season

    combined_data.append(data)

df_all = pd.concat(combined_data, ignore_index=True)

print("\nCOMBINED DATASET:")
print("Total race observations:", len(df_all))

print("\nCOLUMNS:")
print(df_all.columns.tolist())

print("\nOBSERVATIONS BY GENDER AND SEASON:")
print(df_all.groupby(["Gender", "Season"]).size())


# ============================================================
# STEP 3: CONVERT RACE TIMES INTO SECONDS
# ============================================================

def convert_race_time(time):

    if pd.isna(time):
        return float("nan")

    minutes, seconds = str(time).strip().split(":")

    return int(minutes) * 60 + float(seconds)


df_all["Time_seconds"] = (
    df_all["Time (min:sec)"].apply(convert_race_time)
)

df_all["Time_minutes"] = df_all["Time_seconds"] / 60

print("\nRACE TIME CONVERSION:")
print(
    df_all[
        ["Time (min:sec)", "Time_seconds", "Time_minutes"]
    ].head()
)


# ============================================================
# STEP 4: CLEAN WEATHER DATA
# ============================================================

# Remove degree symbols from temperature
df_all["Temp (F)"] = pd.to_numeric(
    df_all["Temp (F)"]
    .astype(str)
    .str.replace("°", "", regex=False)
    .str.strip(),
    errors="coerce"
)

# Convert humidity, wind, and rainfall to numeric values
weather_columns = [
    "Temp (F)",
    "Humidity",
    "Wind (mph)",
    "Rain Racetime (in)",
    "Rain 24hr before (in)"
]

for column in weather_columns:

    df_all[column] = pd.to_numeric(
        df_all[column]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.strip(),
        errors="coerce"
    )

# Convert humidity percentages to fractions if needed
# Existing Excel values like 0.49 already represent 49%.
humidity_percent_mask = df_all["Humidity"] > 1

df_all.loc[humidity_percent_mask, "Humidity"] = (
    df_all.loc[humidity_percent_mask, "Humidity"] / 100
)

# Ensure Grand Prix numbers are numeric
df_all["Grand Prix"] = pd.to_numeric(
    df_all["Grand Prix"],
    errors="coerce"
)

print("\nMISSING VALUES:")
print(df_all.isna().sum())


# ============================================================
# STEP 5: DESCRIPTIVE STATISTICS
# ============================================================

print("\nDESCRIPTIVE STATISTICS:")

print(
    df_all[
        ["Time_seconds"] + weather_columns
    ].describe().round(3)
)


# ============================================================
# STEP 6: ORIGINAL WEATHER CORRELATIONS
# ============================================================

print("\nWEATHER VS. RACE TIME CORRELATIONS:")

original_correlations = df_all[
    ["Time_seconds"] + weather_columns
].corr(numeric_only=True)["Time_seconds"]

print(original_correlations)


# ============================================================
# STEP 7: RUNNER-ADJUSTED WEATHER CORRELATIONS
# ============================================================

runner_groups = [
    "Gender",
    "Season",
    "School",
    "Runner_ID"
]

# Calculate each runner's average finishing time
df_all["Runner_average"] = (
    df_all.groupby(runner_groups)["Time_seconds"]
    .transform("mean")
)

# Compare each race with that runner's average
df_all["Time_difference"] = (
    df_all["Time_seconds"] - df_all["Runner_average"]
)

# Count observations for each runner
df_all["Race_count"] = (
    df_all.groupby(runner_groups)["Time_seconds"]
    .transform("count")
)

# Keep runners who competed in at least two races
repeat_runners = df_all[
    df_all["Race_count"] >= 2
].copy()

print("\nRUNNER-ADJUSTED WEATHER CORRELATIONS:")

adjusted_correlations = repeat_runners[
    weather_columns + ["Time_difference"]
].corr(numeric_only=True)["Time_difference"]

print(adjusted_correlations)

print("\nObservations from runners with multiple races:")
print(len(repeat_runners))

# STEP 7B: Verify excluded observations

race_counts = df_all.groupby(runner_groups).size()

single_race_runners = race_counts[race_counts == 1]

print("\n=== SAMPLE SIZE VERIFICATION ===")

print("Total observations:", len(df_all))

print("Runners with only one race:",
      len(single_race_runners))

print("Observations from single-race runners:",
      single_race_runners.sum())

print("Observations from repeat runners:",
      race_counts[race_counts > 1].sum())

print("Observations excluded:",
      len(df_all) - len(repeat_runners))

# ============================================================
# STEP 8: SIMPLE TEMPERATURE REGRESSION
# ============================================================

regression_data = repeat_runners[
    ["Temp (F)", "Time_difference"]
].dropna()

result = linregress(
    regression_data["Temp (F)"],
    regression_data["Time_difference"]
)

print("\nTEMPERATURE REGRESSION RESULTS:")

print("Slope (seconds per °F):",
      round(result.slope, 3))

print("Correlation:",
      round(result.rvalue, 3))

print("R-squared:",
      round(result.rvalue ** 2, 3))

print("P-value (exploratory):",
      round(result.pvalue, 5))


# ============================================================
# STEP 9: WITHIN-RUNNER TEMPERATURE REGRESSION
# ============================================================

# Calculate each runner's average race temperature
repeat_runners["Runner_avg_temp"] = (
    repeat_runners.groupby(runner_groups)["Temp (F)"]
    .transform("mean")
)

# Compare temperature against each runner's average
repeat_runners["Temp_difference"] = (
    repeat_runners["Temp (F)"]
    - repeat_runners["Runner_avg_temp"]
)

within_data = repeat_runners[
    ["Temp_difference", "Time_difference"]
].dropna()

result_within = linregress(
    within_data["Temp_difference"],
    within_data["Time_difference"]
)

print("\nWITHIN-RUNNER TEMPERATURE REGRESSION:")

print("Slope:",
      round(result_within.slope, 3),
      "seconds per °F")

print("Correlation:",
      round(result_within.rvalue, 3))

print("R-squared:",
      round(result_within.rvalue ** 2, 3))


# ============================================================
# STEP 10: TEMPERATURE VS. GRAND PRIX
# ============================================================

print("\nAVERAGE TEMPERATURE BY GRAND PRIX:")

print(
    df_all.groupby("Grand Prix")["Temp (F)"]
    .agg(["count", "mean", "min", "max"])
    .round(2)
)

print("\nGRAND PRIX VS. TEMPERATURE CORRELATION:")

print(
    df_all[
        ["Grand Prix", "Temp (F)"]
    ].corr()
)


# ============================================================
# STEP 11: MULTIPLE REGRESSION
# CONTROL FOR GRAND PRIX AND RUNNER IDENTITY
# ============================================================

model_data = repeat_runners[
    [
        "Time_seconds",
        "Temp (F)",
        "Grand Prix",
        "Gender",
        "Season",
        "School",
        "Runner_ID"
    ]
].dropna().copy()

# Identify each runner within their season and school
model_data["Runner_key"] = (
    model_data["Gender"].astype(str) + "_" +
    model_data["Season"].astype(str) + "_" +
    model_data["School"].astype(str) + "_" +
    model_data["Runner_ID"].astype(str)
)

model = smf.ols(
    'Time_seconds ~ Q("Temp (F)") + C(Q("Grand Prix")) + C(Runner_key)',
    data=model_data
).fit()

print("\nTEMPERATURE EFFECT AFTER CONTROLLING FOR RACE ORDER:")

print(
    "Temperature slope:",
    round(model.params['Q("Temp (F)")'], 3),
    "seconds per °F"
)

print("\nNumber of observations:", int(model.nobs))


# ============================================================
# STEP 12A: RUNNER-ADJUSTED PERFORMANCE BY GRAND PRIX
# ============================================================

print("\nAVERAGE RUNNER-ADJUSTED TIME BY GRAND PRIX:")

print(
    repeat_runners.groupby("Grand Prix")["Time_difference"]
    .agg(["count", "mean", "median"])
    .round(2)
)


# ============================================================
# STEP 12B: PAIRED COMPARISON OF GP 1 VS. GP 6
# ============================================================

paired_data = df_all[
    df_all["Grand Prix"].isin([1, 6])
].copy()

# Each row represents a runner
# Columns represent GP 1 and GP 6
paired = paired_data.pivot_table(
    index=runner_groups,
    columns="Grand Prix",
    values="Time_seconds",
    aggfunc="mean"
)

# Keep only runners who completed both races
paired = paired.dropna(subset=[1, 6])

# Calculate GP 6 time minus GP 1 time
paired["Time_change"] = paired[6] - paired[1]

print("\nPAIRED GP 1 VS GP 6 ANALYSIS:")

print("Number of paired runners:", len(paired))

print(
    "Average time change:",
    round(paired["Time_change"].mean(), 2),
    "seconds"
)

print(
    "Median time change:",
    round(paired["Time_change"].median(), 2),
    "seconds"
)

print(
    "Percentage who improved:",
    round(
        (paired["Time_change"] < 0).mean() * 100,
        1
    ),
    "%"
)


# ============================================================
# ANALYSIS COMPLETE
# ============================================================

print("\n====================================")
print("QUEENS RUNNERS ANALYSIS COMPLETE!")
print("====================================")

# STEP 13: Compare GP 1 vs GP 6 by season and gender

paired_results = paired.reset_index()

summary = (
    paired_results
    .groupby(["Season", "Gender"])
    .agg(
        Runners=("Time_change", "count"),
        Mean_change=("Time_change", "mean"),
        Median_change=("Time_change", "median"),
        Improved=("Time_change", lambda x: (x < 0).sum())
    )
)

summary["Percent_improved"] = (
    summary["Improved"] / summary["Runners"] * 100
)

print("\nGP 1 VS GP 6 BY SEASON AND GENDER:")
print(summary.round(2))

# STEP 14: Temperature by season and Grand Prix

temperature_summary = (
    df_all.groupby(["Season", "Grand Prix"])["Temp (F)"]
    .agg(["count", "mean"])
    .round(2)
)

print("\nTEMPERATURE BY SEASON AND GRAND PRIX:")
print(temperature_summary)

# ============================================================
# STEP 16: MULTIPLE WEATHER REGRESSION
# Temperature + Humidity + Wind Speed
# Controls: Runner identity and Grand Prix within season
# ============================================================

weather_model_data = repeat_runners[
    [
        "Time_seconds",
        "Temp (F)",
        "Humidity",
        "Wind (mph)",
        "Grand Prix",
        "Gender",
        "Season",
        "School",
        "Runner_ID"
    ]
].dropna().copy()

# Create a unique identifier for each runner
weather_model_data["Runner_key"] = (
    weather_model_data["Gender"].astype(str) + "_" +
    weather_model_data["Season"].astype(str) + "_" +
    weather_model_data["School"].astype(str) + "_" +
    weather_model_data["Runner_ID"].astype(str)
)

# Multiple linear regression
weather_model = smf.ols(
    'Time_seconds ~ Q("Temp (F)")'
    ' + Humidity'
    ' + Q("Wind (mph)")'
    ' + C(Season):C(Q("Grand Prix"))'
    ' + C(Runner_key)',
    data=weather_model_data
).fit()

print("\nMULTIPLE WEATHER REGRESSION RESULTS:")

for variable in [
    'Q("Temp (F)")',
    'Humidity',
    'Q("Wind (mph)")'
]:
    print(
        variable,
        "Slope:",
        round(weather_model.params[variable], 3)
    )

print("R-squared:", round(weather_model.rsquared, 3))
print("\nObservations:", int(weather_model.nobs))

# STEP 17: Compare models with and without weather

# Model A: No weather variables
baseline_model = smf.ols(
    'Time_seconds ~ C(Season):C(Q("Grand Prix"))'
    ' + C(Runner_key)',
    data=weather_model_data
).fit()

# Model B: Includes weather variables
# Already calculated as weather_model

print("\nMODEL COMPARISON:")

print("Baseline R-squared:",
      round(baseline_model.rsquared, 4))

print("Weather model R-squared:",
      round(weather_model.rsquared, 4))

print("Increase in R-squared:",
      round(
          weather_model.rsquared - baseline_model.rsquared,
          4
      ))

print("\nADJUSTED R-SQUARED:")

print("Baseline:",
      round(baseline_model.rsquared_adj, 4))

print("Weather model:",
      round(weather_model.rsquared_adj, 4))
print("Baseline R²:", round(baseline_model.rsquared, 6))
print("Weather R²:", round(weather_model.rsquared, 6))

print(
    "R² increase:",
    round(weather_model.rsquared - baseline_model.rsquared, 6)
)

# STEP 18: Check weather variation within races

race_weather = (
    weather_model_data.groupby(
        ["Season", "Gender", "Grand Prix"]
    )
    .agg(
        Runners=("Time_seconds", "count"),
        Unique_temps=("Temp (F)", "nunique"),
        Unique_humidity=("Humidity", "nunique"),
        Unique_wind=("Wind (mph)", "nunique")
    )
)

print("\nWEATHER VARIATION WITHIN RACES:")
print(race_weather.to_string())

print(
    "\nNumber of season-gender-GP groups:",
    len(race_weather)
)

# Check distinct weather combinations within each group

weather_combinations = (
    weather_model_data.groupby(
        ["Season", "Gender", "Grand Prix"]
    )[["Temp (F)", "Humidity", "Wind (mph)"]]
    .apply(lambda group: group.drop_duplicates())
)

print(weather_combinations.to_string())


# STEP 19: Count distinct race dates

date_column = [
    col for col in df_all.columns
    if "Date" in col
][0]

df_all["Race_date"] = pd.to_datetime(
    df_all[date_column],
    errors="coerce"
).dt.date

print("\nTOTAL UNIQUE RACE DATES:")
print(df_all["Race_date"].nunique())

print("\nRACE DATES BY SEASON AND GENDER:")
print(
    df_all.groupby(["Season", "Gender"])["Race_date"]
    .nunique()
)

print("\nMISSING OR UNREADABLE DATES:")
print(df_all["Race_date"].isna().sum())

# STEP 20: Final race-date schedule

race_schedule = (
    df_all[
        ["Season", "Gender", "Grand Prix", "Race_date"]
    ]
    .drop_duplicates()
    .sort_values(
        ["Season", "Grand Prix", "Gender"]
    )
)

print("\nFINAL RACE SCHEDULE:")
print(race_schedule.to_string(index=False))

print("\nNUMBER OF RACE GROUPS:")
print(len(race_schedule))

# ============================================================
# STEP 21: CLUSTER-ROBUST STANDARD ERRORS
# Account for runners sharing the same race date
# ============================================================

# Identify the original date column
date_column = [
    col for col in df_all.columns
    if "Date" in col
][0]

# Create a dataset with the same observations as
# our multiple weather regression
cluster_data = weather_model_data.copy()

# Match dates using the original DataFrame indices
cluster_data["Race_date"] = pd.to_datetime(
    df_all.loc[cluster_data.index, date_column],
    errors="coerce"
).dt.date

print("\nMISSING RACE DATES:")
print(cluster_data["Race_date"].isna().sum())

# Refit the same regression with clustered standard errors
cluster_model = smf.ols(
    'Time_seconds ~ Q("Temp (F)")'
    ' + Humidity'
    ' + Q("Wind (mph)")'
    ' + C(Season):C(Q("Grand Prix"))'
    ' + C(Runner_key)',
    data=cluster_data
).fit(
    cov_type="cluster",
    cov_kwds={
        "groups": cluster_data["Race_date"]
    }
)

print("\nCLUSTER-ROBUST WEATHER REGRESSION:")

for variable in [
    'Q("Temp (F)")',
    'Humidity',
    'Q("Wind (mph)")'
]:
    print("\n", variable)
    print("Coefficient:",
          round(cluster_model.params[variable], 3))
    print("Standard error:",
          round(cluster_model.bse[variable], 3))
    print("P-value:",
          round(cluster_model.pvalues[variable], 4))

print("\nNumber of race-date clusters:",
      cluster_data["Race_date"].nunique())

# STEP 22: Verify the regression coefficients match

variables = [
    'Q("Temp (F)")',
    'Humidity',
    'Q("Wind (mph)")'
]

print("\nCOEFFICIENT COMPARISON:")

for variable in variables:
    print(
        variable,
        "| Original:",
        round(weather_model.params[variable], 4),
        "| Clustered:",
        round(cluster_model.params[variable], 4)
    )

print("\nORIGINAL OBSERVATIONS:", int(weather_model.nobs))
print("CLUSTERED OBSERVATIONS:", int(cluster_model.nobs))

# STEP 23: Diagnose the clustered covariance matrix

import numpy as np

# Check standard errors across all model parameters
standard_errors = cluster_model.bse

print("\nTOTAL MODEL PARAMETERS:")
print(len(standard_errors))

print("\nINVALID STANDARD ERRORS:")
print("NaN:", standard_errors.isna().sum())
print("Infinite:", np.isinf(standard_errors).sum())

print("\nNUMBER OF DATE CLUSTERS:")
print(cluster_data["Race_date"].nunique())

print("\nWEATHER COEFFICIENT CONFIDENCE INTERVALS:")

weather_variables = [
    'Q("Temp (F)")',
    'Humidity',
    'Q("Wind (mph)")'
]

print(
    cluster_model.conf_int().loc[
        weather_variables
    ]
)


# STEP 25: Check model rank and residual degrees of freedom

import numpy as np

X = cluster_model.model.exog

print("\nMODEL DIAGNOSTICS:")

print("Observations:", X.shape[0])
print("Design matrix columns:", X.shape[1])
print("Matrix rank:", np.linalg.matrix_rank(X))

print(
    "Full rank:",
    np.linalg.matrix_rank(X) == X.shape[1]
)

print(
    "Residual degrees of freedom:",
    cluster_model.df_resid
)

print(
    "Condition number:",
    cluster_model.condition_number
)

# STEP 24: Build a full-rank fixed-effects model

import numpy as np
import pandas as pd
import statsmodels.api as sm
import patsy

clean_model_data = cluster_data.copy()

# Identify each season-specific Grand Prix event
clean_model_data["Race_event"] = (
    clean_model_data["Season"].astype(str)
    + "_GP"
    + clean_model_data["Grand Prix"].astype(str)
)

formula = (
    'Time_seconds ~ Q("Temp (F)")'
    ' + Humidity'
    ' + Q("Wind (mph)")'
    ' + C(Race_event)'
    ' + C(Runner_key)'
)

# Build the design matrix
y, X = patsy.dmatrices(
    formula,
    data=clean_model_data,
    return_type="dataframe"
)

# Remove linearly dependent columns using
# pivoted QR decomposition
from scipy.linalg import qr

Q, R, pivots = qr(
    X.to_numpy(),
    mode="economic",
    pivoting=True
)

# STEP 24A: Create the full-rank design matrix

diagonal = np.abs(np.diag(R))

tolerance = (
    max(X.shape)
    * np.finfo(float).eps
    * diagonal.max()
)

rank = int(np.sum(diagonal > tolerance))

selected_columns = sorted(pivots[:rank])

X_full_rank = X.iloc[:, selected_columns]

print("\nORIGINAL COLUMNS:", X.shape[1])
print("FULL-RANK COLUMNS:", X_full_rank.shape[1])
print("FINAL MATRIX RANK:",
      np.linalg.matrix_rank(X_full_rank))


# STEP 24B: Verify removed regression columns

removed_columns = [
    col for col in X.columns
    if col not in X_full_rank.columns
]

print("\n=== REMOVED REGRESSION COLUMNS ===")
print(removed_columns)

print("\nNumber removed:", len(removed_columns))

print("\nWeather predictors retained:")

for variable in [
    'Q("Temp (F)")',
    'Humidity',
    'Q("Wind (mph)")'
]:
    print(variable, variable in X_full_rank.columns)

# STEP 26: Refit the full-rank regression model

revised_model = sm.OLS(
    y,
    X_full_rank
).fit()

print("\nREVISED WEATHER COEFFICIENTS:")

for variable in weather_variables:
    if variable in revised_model.params:
        print(
            variable,
            round(revised_model.params[variable], 4)
        )
    else:
        print(
            variable,
            "Not retained in the independent column set"
        )


# STEP 27: Cluster-robust inference for the full-rank model

revised_cluster_model = sm.OLS(
    y,
    X_full_rank
).fit(
    cov_type="cluster",
    cov_kwds={
        "groups": clean_model_data.loc[
            X_full_rank.index,
            "Race_date"
        ]
    }
)

print("\nREVISED CLUSTER-ROBUST RESULTS:")

for variable in weather_variables:
    if variable in revised_cluster_model.params:
        print("\n", variable)
        print(
            "Coefficient:",
            round(revised_cluster_model.params[variable], 4)
        )
        print(
            "Standard error:",
            round(revised_cluster_model.bse[variable], 4)
        )
        print(
            "P-value:",
            round(revised_cluster_model.pvalues[variable], 4)
        )

print("\nUNDEFINED STANDARD ERRORS:")
print(revised_cluster_model.bse.isna().sum())

print("\nDESIGN MATRIX FULL RANK:")
print(
    np.linalg.matrix_rank(X_full_rank)
    == X_full_rank.shape[1]
)

# STEP 28: Final model verification

print("\n=== FINAL MODEL VERIFICATION ===")

final_columns = X_full_rank.shape[1]
final_rank = np.linalg.matrix_rank(X_full_rank)

print("Observations:", X_full_rank.shape[0])
print("Number of columns:", final_columns)
print("Matrix rank:", final_rank)

print("Full rank:", final_rank == final_columns)

print("Number of race-date clusters:",
      clean_model_data.loc[
          X_full_rank.index, "Race_date"
      ].nunique())

print("Undefined standard errors:",
      revised_cluster_model.bse.isna().sum())

# ============================================================
# STEP 29: VERIFY R-SQUARED MODEL COMPARISON
# ============================================================

# The final model already includes weather predictors
full_model = sm.OLS(y, X_full_rank).fit()

# Identify the three weather predictor columns
weather_columns = [
    'Q("Temp (F)")',
    'Humidity',
    'Q("Wind (mph)")'
]

# Create a baseline model by removing ONLY weather predictors
X_baseline = X_full_rank.drop(columns=weather_columns)

# Fit the baseline model using the same observations
baseline_model = sm.OLS(y, X_baseline).fit()

# Compare explanatory power
print("\n=== FINAL R-SQUARED COMPARISON ===")
print("Observations:", len(y))
print("Without weather predictors:", round(baseline_model.rsquared, 6))
print("With weather predictors:", round(full_model.rsquared, 6))
print(
    "R-squared difference:",
    round(full_model.rsquared - baseline_model.rsquared, 6)
)
