# importing packages
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats
import statsmodels.api as sm
import sklearn.metrics
from sklearn.metrics import r2_score

# importing data locally from computer; see README for links to datasets and cleaning-data.py for data cleaning steps
house_data = pd.read_csv("house_data.csv")
sen_data = pd.read_csv("sen_data.csv")

# defining a function for a post-hoc test
def post_hoc(obs_df, exp_df): 
  result = pd.DataFrame()
  for r in obs_df.index: 
    for c in obs_df.columns: 
      observed = obs_df.loc[r, c]
      expected = exp_df.loc[r, c]
      col_sum = obs_df[c].sum()
      row_sum = obs_df.loc[r, :].sum()
      exp_row_prop = expected / row_sum
      exp_col_prop = expected / col_sum
      std_resid = (observed - expected) / (math.sqrt(expected * (1 - exp_row_prop) * (1 - exp_col_prop)))
      p_val = scipy.stats.norm.sf(std_resid)
      result.loc[r, c] = p_val
      if p_val < 0.05 / 15: 
        print("{}, {}: {}".format(r, c, p_val))
  return result

# creating heatmaps—before using R, I made these to see the collinearity among different variables to prevent multicollinearity in my regression model.
sns.heatmap(data=house_data.corr(), vmin=-1, vmax=1)
sns.heatmap(data=sen_data.corr(), vmin=-1, vmax=1)

# I used the same statistical tests for the Senate and House.
# The code for the test on the House data comes first, so I will comment that with more detail than the Senate, which is close to a repeat of the same code.

# subsets found by R model, in ascending order of R^2 value
# I used an R program to run best subsets regression after I found it difficult to find the best variables to use for the linear regression by hand. I used
# R's expanded functionality to make the model instead. The R file is in this repository, in a file called `best-subsets.R`.
subsets = {
    1: ["total_contrib"],
    2: ["other_pac_percent"],
    3: ["OTHER_POL_CMTE_CONTRIB"],
    4: ["TTL_RECEIPTS", "other_pac_percent"], # ok
    5: ["TTL_INDIV_CONTRIB", "other_pac_percent"], # ok
    6: ["total_contrib", "other_pac_percent"], # ok
    7: ["COH_BOP", "total_contrib", "other_pac_percent"], # ok
    8: ["COH_COP", "TTL_INDIV_CONTRIB", "other_pac_percent"], # ok
    9: ["COH_COP", "total_contrib", "other_pac_percent"], # ok
    10: ["COH_COP", "total_contrib", "pol_pty_percent", "other_pac_percent"], # ok
    11: ["COH_COP", "total_contrib", "indiv_percent", "other_pac_percent"],
    12: ["COH_COP", "total_contrib", "indiv_percent", "pol_pty_percent"],
    13: ["TRANS_FROM_AUTH", "COH_COP", "total_contrib", "pol_pty_percent", "other_pac_percent"],
    14: ["TRANS_FROM_AUTH", "COH_COP", "total_contrib", "indiv_percent", "other_pac_percent"],
    15: ["TRANS_FROM_AUTH", "COH_COP", "total_contrib", "indiv_percent", "pol_pty_percent"],
    16: ["TRANS_FROM_AUTH", "COH_COP", "CAND_CONTRIB", "total_contrib", "pol_pty_percent", "other_pac_percent"],
    17: ["TRANS_FROM_AUTH", "COH_COP", "CAND_CONTRIB", "total_contrib", "indiv_percent", "pol_pty_percent"],
    18: ["TRANS_FROM_AUTH", "COH_COP", "CAND_CONTRIB", "total_contrib", "indiv_percent", "other_pac_percent"],
    19: ["TTL_RECEIPTS", "COH_COP", "CAND_CONTRIB", "CAND_LOANS", "total_contrib", "indiv_percent", "other_pac_percent"],
    20: ["TTL_RECEIPTS", "COH_COP", "CAND_CONTRIB", "CAND_LOANS", "total_contrib", "indiv_percent", "pol_pty_percent"],
    21: ["TTL_RECEIPTS", "COH_COP", "CAND_CONTRIB", "CAND_LOANS", "total_contrib", "pol_pty_percent", "other_pac_percent"],
    22: ["TTL_RECEIPTS", "COH_COP", "CAND_CONTRIB", "CAND_LOANS", "OTHER_LOANS", "total_contrib", "indiv_percent", "other_pac_percent"],
    23: ["TTL_RECEIPTS", "COH_COP", "CAND_CONTRIB", "CAND_LOANS", "OTHER_LOANS", "total_contrib", "indiv_percent", "pol_pty_percent"],
    24: ["TTL_RECEIPTS", "COH_COP", "CAND_CONTRIB", "CAND_LOANS", "OTHER_LOANS", "total_contrib", "pol_pty_percent", "other_pac_percent"],
    25: ["TTL_RECEIPTS", "TRANS_FROM_AUTH", "COH_COP", "CAND_CONTRIB", "CAND_LOANS", "OTHER_LOANS", "total_contrib", "indiv_percent", "other_pac_percent"],
    26: ["TTL_RECEIPTS", "TRANS_FROM_AUTH", "COH_COP", "CAND_CONTRIB", "CAND_LOANS", "OTHER_LOANS", "total_contrib", "indiv_percent", "pol_pty_percent"],
    27: ["TTL_RECEIPTS", "TRANS_FROM_AUTH", "COH_COP", "CAND_CONTRIB", "CAND_LOANS", "OTHER_LOANS", "total_contrib", "pol_pty_percent", "other_pac_percent"]
}

# In order to visualize the accuracies of the models created by each subset, I ran a linear regression on each subset and graphed the mean squared errors of each
# as an extra step to compare the different models.

# splitting data
Xy = house_data[["CAND_ICI", "party", "TTL_RECEIPTS", "TRANS_FROM_AUTH", "TTL_DISB", "TRANS_TO_AUTH",
       "COH_BOP", "COH_COP", "CAND_CONTRIB", "CAND_LOANS", "OTHER_LOANS", "CAND_LOAN_REPAY",
       "OTHER_LOAN_REPAY", "DEBTS_OWED_BY", "TTL_INDIV_CONTRIB", "OTHER_POL_CMTE_CONTRIB",
       "POL_PTY_CONTRIB", "INDIV_REFUNDS", "CMTE_REFUNDS", "candidatevotes",
       "totalvotes", "total_contrib", "indiv_percent", "pol_pty_percent",
       "other_pac_percent", "usd_per_vote", "usd_per_percent"]].dropna().reset_index(drop=True)
train, test = np.split(Xy.sample(frac=1), [int(0.8 * Xy.shape[0])])

models_df = pd.DataFrame()

# creating models
for i in subsets: 
  X_train = train[subsets[i]]
  y_train = train["candidatevotes"] / train["totalvotes"]
  X_test = test[subsets[i]]
  y_test = test["candidatevotes"] / test["totalvotes"]
  model = sm.OLS(y_train, X_train)
  model = model.fit()
  models_df.loc[i - 1, "subset"] = i
  models_df.loc[i - 1,"summary"] = model.summary
  models_df.loc[i - 1, "train_mse"] = sklearn.metrics.mean_squared_error(y_train, model.predict(X_train))
  models_df.loc[i - 1, "test_mse"] = sklearn.metrics.mean_squared_error(y_test, model.predict(X_test))

# plotting the accuracies
plt.figure(figsize=(15, 4))
plt.plot(models_df["subset"], models_df["train_mse"], label="Train MSE")
plt.plot(models_df["subset"], models_df["test_mse"], label="Test MSE")
plt.xticks(range(1, 28))
plt.xlabel("Subset of Variables")
plt.ylabel("Mean Squared Error")
plt.title("Linear Regression — House")
plt.legend()

# the best subsets, found with R and python working together
house_preds = {
    1: ["TTL_RECEIPTS", "other_pac_percent"],
    2: ["TTL_INDIV_CONTRIB", "other_pac_percent"],
    3: ["total_contrib", "other_pac_percent"]
}

# setting up X and Y datasets for the statistical tests
house_data = house_data[["CAND_ICI", "party", "TTL_RECEIPTS", "TRANS_FROM_AUTH", "TTL_DISB", "TRANS_TO_AUTH",
       "COH_BOP", "COH_COP", "CAND_CONTRIB", "CAND_LOANS", "OTHER_LOANS", "CAND_LOAN_REPAY",
       "OTHER_LOAN_REPAY", "DEBTS_OWED_BY", "TTL_INDIV_CONTRIB", "OTHER_POL_CMTE_CONTRIB",
       "POL_PTY_CONTRIB", "INDIV_REFUNDS", "CMTE_REFUNDS", "candidatevotes",
       "totalvotes", "total_contrib", "indiv_percent", "pol_pty_percent",
       "other_pac_percent", "usd_per_vote", "usd_per_percent"]].dropna().reset_index(drop=True)

# these datasets are for the categorical data, which are used in the chi-squared tests.
house_i = house_data[house_data["CAND_ICI"] == "I"]
house_c = house_data[house_data["CAND_ICI"] == "C"]
house_o = house_data[house_data["CAND_ICI"] == "O"]

house_dem = house_data[house_data["party"] == "DEM"]
house_rep = house_data[house_data["party"] == "REP"]
house_third = house_data[(house_data["party"] != "REP") & (house_data["party"] != "DEM")]

# running the linear regression models to find relationships between numerical variables of campaign finance and election success
for i in range(1, 4): 
  train, test = np.split(house_data.sample(frac=1), [int(0.8 * house_data.shape[0])])
  X_train = train[house_preds[i]]
  y_train = train["candidatevotes"] / train["totalvotes"]
  X_test = test[house_preds[i]]
  y_test = test["candidatevotes"] / test["totalvotes"]
  model = sm.OLS(y_train, X_train)
  model = model.fit()
  print(model.summary())
  print(sklearn.metrics.mean_squared_error(y_test, model.predict(X_test)))
  print()

# running the chi-squared tests for political party
# This test would show what pairs of variables showed statistical significance; in other words, which factors were related to political party.
for var in ["indiv_percent", "pol_pty_percent", "other_pac_percent"]: 
  chi_df = pd.concat([house_data["party"], house_data[var]], axis=1)
  chi_df["count"] = 1
  if var == "indiv_percent": 
    print("Percent of Contributions from Individuals")
  elif var == "pol_pty_percent": 
    print("Percent of Contributions from Political Parties")
  else: 
    print("Percent of Contributions from PACs")
  for i in range(chi_df.shape[0]): 
    if chi_df.loc[i, var] <= 0.2: 
      chi_df.loc[i, var] = 2
    elif chi_df.loc[i, var] <= 0.4: 
      chi_df.loc[i, var] = 4
    elif chi_df.loc[i, var] <= 0.6: 
      chi_df.loc[i, var] = 6
    elif chi_df.loc[i, var] <= 0.8: 
      chi_df.loc[i, var] = 8
    else: 
      chi_df.loc[i, var] = 10
  chi_df[var] = chi_df[var].map({2: "0.0-0.2", 4: "0.2-0.4", 6: "0.4-0.6", 8: "0.6-0.8", 10: "0.8-1.0"})
  chi_df = chi_df.groupby(by=["party", var], as_index=False).sum()
  chi_df = chi_df.pivot(index=var, columns="party", values="count")
  chi_df = chi_df.fillna(0)
  chi2, p, dof, ex = scipy.stats.chi2_contingency(chi_df, correction=False)
  print("P-value: {}".format(p))
  exp_df = pd.DataFrame(ex)
  exp_df = exp_df.rename(columns={0: "DEM", 1: "IND", 2: "LIB", 3: "OTH", 4: "REP"})
  exp_df = exp_df.set_index(chi_df.index)
  display(post_hoc(chi_df, exp_df))
  print()
  
# another chi-squared test, this time with the percentage of vote won instead of sources of campaign fincance
pivot_df = pd.concat([house_data["party"], house_data["candidatevotes"] / house_data["totalvotes"]], axis=1)
pivot_df = pivot_df.rename({"party": "party", 0: "vote_percent"}, axis=1)
pivot_df["count"] = 1
for i in range(pivot_df.shape[0]): 
  if pivot_df.loc[i, "vote_percent"] <= 0.2: 
    pivot_df.loc[i, "vote_percent"] = 2
  elif pivot_df.loc[i, "vote_percent"] <= 0.4: 
    pivot_df.loc[i, "vote_percent"] = 4
  elif pivot_df.loc[i, "vote_percent"] <= 0.6: 
    pivot_df.loc[i, "vote_percent"] = 6
  elif pivot_df.loc[i, "vote_percent"] <= 0.8: 
    pivot_df.loc[i, "vote_percent"] = 8
  else: 
    pivot_df.loc[i, "vote_percent"] = 10
pivot_df["vote_percent"] = pivot_df["vote_percent"].map({2: "0.0-0.2", 4: "0.2-0.4", 6: "0.4-0.6", 8: "0.6-0.8", 10: "0.8-1.0"})
pivot_df = pivot_df.groupby(by=["party", "vote_percent"], as_index=False).sum()
pivot_df = pivot_df.pivot(index="vote_percent", columns="party", values="count")
pivot_df = pivot_df.fillna(0)
chi2, p, dof, ex = scipy.stats.chi2_contingency(pivot_df, correction=False)
print(p)

# post hoc test: shows how different pairs are related to each other
obs_df = pivot_df
exp_df = pd.DataFrame(ex)
exp_df.set_index([["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]], inplace=True)
exp_df = exp_df.rename(columns={0: "DEM", 1: "IND", 2: "LIB", 3: "OTH", 4: "REP"})

display(post_hoc(obs_df, exp_df))

# same tests as above with incumbency instead of political party
for var in ["indiv_percent", "pol_pty_percent", "other_pac_percent"]: 
  chi_df = pd.concat([house_data["CAND_ICI"], house_data[var]], axis=1)
  chi_df["count"] = 1
  if var == "indiv_percent": 
    print("Percent of Contributions from Individuals")
  elif var == "pol_pty_percent": 
    print("Percent of Contributions from Political Parties")
  else: 
    print("Percent of Contributions from PACs")
  for i in range(chi_df.shape[0]): 
    if chi_df.loc[i, var] <= 0.2: 
      chi_df.loc[i, var] = 2
    elif chi_df.loc[i, var] <= 0.4: 
      chi_df.loc[i, var] = 4
    elif chi_df.loc[i, var] <= 0.6: 
      chi_df.loc[i, var] = 6
    elif chi_df.loc[i, var] <= 0.8: 
      chi_df.loc[i, var] = 8
    else: 
      chi_df.loc[i, var] = 10
  chi_df[var] = chi_df[var].map({2: "0.0-0.2", 4: "0.2-0.4", 6: "0.4-0.6", 8: "0.6-0.8", 10: "0.8-1.0"})
  chi_df = chi_df.groupby(by=["CAND_ICI", var], as_index=False).sum()
  chi_df = chi_df.pivot(index=var, columns="CAND_ICI", values="count")
  chi_df = chi_df.fillna(0)
  chi2, p, dof, ex = scipy.stats.chi2_contingency(chi_df, correction=False)
  print("P-value: {}".format(p))
  exp_df = pd.DataFrame(ex)
  exp_df = exp_df.rename(columns={0: "C", 1: "I", 2: "O"})
  exp_df = exp_df.set_index(chi_df.index)
  display(post_hoc(chi_df, exp_df))
  print()

# chi2 test
pivot_df = pd.concat([house_data["CAND_ICI"], house_data["candidatevotes"] / house_data["totalvotes"]], axis=1)
pivot_df = pivot_df.rename({"CAND_ICI": "CAND_ICI", 0: "vote_percent"}, axis=1)
pivot_df["count"] = 1
for i in range(pivot_df.shape[0]): 
  if pivot_df.loc[i, "vote_percent"] <= 0.2: 
    pivot_df.loc[i, "vote_percent"] = 2
  elif pivot_df.loc[i, "vote_percent"] <= 0.4: 
    pivot_df.loc[i, "vote_percent"] = 4
  elif pivot_df.loc[i, "vote_percent"] <= 0.6: 
    pivot_df.loc[i, "vote_percent"] = 6
  elif pivot_df.loc[i, "vote_percent"] <= 0.8: 
    pivot_df.loc[i, "vote_percent"] = 8
  else: 
    pivot_df.loc[i, "vote_percent"] = 10
pivot_df["vote_percent"] = pivot_df["vote_percent"].map({2: "0.0-0.2", 4: "0.2-0.4", 6: "0.4-0.6", 8: "0.6-0.8", 10: "0.8-1.0"})
pivot_df = pivot_df.groupby(by=["CAND_ICI", "vote_percent"], as_index=False).sum()
pivot_df = pivot_df.pivot(index="vote_percent", columns="CAND_ICI", values="count")
pivot_df = pivot_df.fillna(0)
chi2, p, dof, ex = scipy.stats.chi2_contingency(pivot_df, correction=False)
print(p)

# post hoc
obs_df = pivot_df
exp_df = pd.DataFrame(ex)
exp_df.set_index([["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]], inplace=True)
exp_df = exp_df.rename(columns={0: "C", 1: "I", 2: "O"})

display(post_hoc(obs_df, exp_df))

# statistical tests with the Senate
# picking variables
subsets = {
    1: ["COH_COP"],
    2: ["COH_COP", "pol_pty_percent"],
    3: ["COH_COP", "DEBTS_OWED_BY", "pol_pty_percent"],
    4: ["COH_COP", "CAND_CONTRIB", "DEBTS_OWED_BY", "pol_pty_percent"],
    5: ["indiv_percent", "TTL_DISB", "COH_COP", "TTL_INDIV_CONTRIB", "pol_pty_percent"], #COH_COP was originally COH_BOP
    6: ["TTL_RECEIPTS", "TTL_DISB", "COH_COP", "indiv_percent", "pol_pty_percent"],
    7: ["TTL_DISB", "COH_COP", "TTL_INDIV_CONTRIB", "OTHER_POL_CMTE_CONTRIB", "total_contrib", "indiv_percent", ],#"CMTE_REFUNDS", "pol_pty_percent"
    8: ["COH_COP", "CAND_CONTRIB", "TTL_INDIV_CONTRIB", "OTHER_POL_CMTE_CONTRIB", "other_pac_percent"],#"CMTE_REFUNDS", "TTL_DISB", "indiv_percent"],
    9: ["COH_COP", "CAND_CONTRIB", "DEBTS_OWED_BY", "TTL_INDIV_CONTRIB", "pol_pty_percent", "other_pac_percent"]#"CMTE_REFUNDS", "TTL_DISB",  "OTHER_POL_CMTE_CONTRIB", 
}

sen_data = sen_data[["CAND_ICI", "party", "TTL_RECEIPTS", "TRANS_FROM_AUTH", "TTL_DISB", "TRANS_TO_AUTH",
       "COH_BOP", "COH_COP", "CAND_CONTRIB", "CAND_LOANS", "OTHER_LOANS", "CAND_LOAN_REPAY",
       "OTHER_LOAN_REPAY", "DEBTS_OWED_BY", "TTL_INDIV_CONTRIB", "OTHER_POL_CMTE_CONTRIB",
       "POL_PTY_CONTRIB", "INDIV_REFUNDS", "CMTE_REFUNDS", "candidatevotes",
       "totalvotes", "total_contrib", "indiv_percent", "pol_pty_percent",
       "other_pac_percent", "usd_per_vote", "usd_per_percent"]].dropna().reset_index(drop=True)
train, test = np.split(sen_data.sample(frac=1), [int(0.8 * sen_data.shape[0])])

models_df = pd.DataFrame()

for i in subsets: 
  X_train = train[subsets[i]]
  y_train = train["candidatevotes"] / train["totalvotes"]
  X_test = test[subsets[i]]
  y_test = test["candidatevotes"] / test["totalvotes"]
  model = sm.OLS(y_train, X_train)
  model = model.fit()
  models_df.loc[i - 1, "subset"] = i
  models_df.loc[i - 1,"summary"] = model.summary
  models_df.loc[i - 1, "train_mse"] = sklearn.metrics.mean_squared_error(y_train, model.predict(X_train))
  models_df.loc[i - 1, "test_mse"] = sklearn.metrics.mean_squared_error(y_test, model.predict(X_test))
  print(i)
  print(model.summary())
  print()

# plotting accuracies
plt.figure(figsize=(15, 4))
plt.plot(models_df["subset"], models_df["train_mse"], label="Train MSE")
plt.plot(models_df["subset"], models_df["test_mse"], label="Test MSE")
plt.xticks(range(1, 10))
plt.xlabel("Subset of Variables")
plt.ylabel("Mean Squared Error")
plt.title("Linear Regression — Senate")
plt.legend()

sen_preds = {
    1: ["TTL_DISB", "COH_COP", "TTL_INDIV_CONTRIB", "OTHER_POL_CMTE_CONTRIB", "total_contrib", "indiv_percent"],
    2: ["COH_COP", "CAND_CONTRIB", "TTL_INDIV_CONTRIB", "OTHER_POL_CMTE_CONTRIB", "other_pac_percent"],
    3: ["COH_COP", "CAND_CONTRIB", "DEBTS_OWED_BY", "TTL_INDIV_CONTRIB", "pol_pty_percent", "other_pac_percent"]
}

# setting up X, Y, data for chi-squared tests
sen_data = sen_data[["CAND_ICI", "party", "TTL_RECEIPTS", "TRANS_FROM_AUTH", "TTL_DISB", "TRANS_TO_AUTH",
       "COH_BOP", "COH_COP", "CAND_CONTRIB", "CAND_LOANS", "OTHER_LOANS", "CAND_LOAN_REPAY",
       "OTHER_LOAN_REPAY", "DEBTS_OWED_BY", "TTL_INDIV_CONTRIB", "OTHER_POL_CMTE_CONTRIB",
       "POL_PTY_CONTRIB", "INDIV_REFUNDS", "CMTE_REFUNDS", "candidatevotes",
       "totalvotes", "total_contrib", "indiv_percent", "pol_pty_percent",
       "other_pac_percent", "usd_per_vote", "usd_per_percent"]].dropna().reset_index(drop=True)

sen_i = sen_data[sen_data["CAND_ICI"] == "I"]
sen_c = sen_data[sen_data["CAND_ICI"] == "C"]
sen_o = sen_data[sen_data["CAND_ICI"] == "O"]

sen_dem = sen_data[sen_data["party"] == "DEM"]
sen_rep = sen_data[sen_data["party"] == "REP"]
sen_third = sen_data[(sen_data["party"] != "REP") & (sen_data["party"] != "DEM")]

# linear regression
sen_preds_repl = {
    1: ["TTL_DISB", "COH_COP", "TTL_INDIV_CONTRIB", "OTHER_POL_CMTE_CONTRIB", "total_contrib", "other_pac_percent"],
    2: ["COH_COP", "CAND_CONTRIB", "TTL_INDIV_CONTRIB", "OTHER_POL_CMTE_CONTRIB", "indiv_percent"],
    3: ["COH_COP", "CAND_CONTRIB", "DEBTS_OWED_BY", "TTL_INDIV_CONTRIB", "pol_pty_percent", "indiv_percent"]
}

for i in sen_preds_repl: 
  train, test = np.split(sen_data.sample(frac=1), [int(0.8 * sen_data.shape[0])])
  X_train = train[sen_preds_repl[i]]
  y_train = train["candidatevotes"] / train["totalvotes"]
  X_test = test[sen_preds_repl[i]]
  y_test = test["candidatevotes"] / test["totalvotes"]
  model = sm.OLS(y_train, X_train)
  model = model.fit()
  print(model.summary())
  print(sklearn.metrics.mean_squared_error(y_test, model.predict(X_test)))

for i in sen_preds: 
  train, test = np.split(sen_data.sample(frac=1), [int(0.8 * sen_data.shape[0])])
  X_train = train[sen_preds[i]]
  y_train = train["candidatevotes"] / train["totalvotes"]
  X_test = test[sen_preds[i]]
  y_test = test["candidatevotes"] / test["totalvotes"]
  model = sm.OLS(y_train, X_train)
  model = model.fit()
  print(model.summary())
  print(sklearn.metrics.mean_squared_error(y_test, model.predict(X_test)))

# chi-squared tests for political party
pivot_df = pd.concat([sen_data["party"], sen_data["candidatevotes"] / sen_data["totalvotes"]], axis=1)
pivot_df = pivot_df.rename({"party": "party", 0: "vote_percent"}, axis=1)
pivot_df["count"] = 1
for i in range(pivot_df.shape[0]): 
  if pivot_df.loc[i, "vote_percent"] <= 0.2: 
    pivot_df.loc[i, "vote_percent"] = 2
  elif pivot_df.loc[i, "vote_percent"] <= 0.4: 
    pivot_df.loc[i, "vote_percent"] = 4
  elif pivot_df.loc[i, "vote_percent"] <= 0.6: 
    pivot_df.loc[i, "vote_percent"] = 6
  elif pivot_df.loc[i, "vote_percent"] <= 0.8: 
    pivot_df.loc[i, "vote_percent"] = 8
  else: 
    pivot_df.loc[i, "vote_percent"] = 10
pivot_df["vote_percent"] = pivot_df["vote_percent"].map({2: "0.0-0.2", 4: "0.2-0.4", 6: "0.4-0.6", 8: "0.6-0.8", 10: "0.8-1.0"})
pivot_df = pivot_df.groupby(by=["party", "vote_percent"], as_index=False).sum()
pivot_df = pivot_df.pivot(index="vote_percent", columns="party", values="count")
pivot_df = pivot_df.fillna(0)
chi2, p, dof, ex = scipy.stats.chi2_contingency(pivot_df, correction=False)
print(p)

for var in ["indiv_percent", "pol_pty_percent", "other_pac_percent"]: 
  chi_df = pd.concat([sen_data["party"], sen_data[var]], axis=1)
  chi_df["count"] = 1
  if var == "indiv_percent": 
    print("Percent of Contributions from Individuals")
  elif var == "pol_pty_percent": 
    print("Percent of Contributions from Political Parties")
  else: 
    print("Percent of Contributions from PACs")
  for i in range(chi_df.shape[0]): 
    if chi_df.loc[i, var] <= 0.2: 
      chi_df.loc[i, var] = 2
    elif chi_df.loc[i, var] <= 0.4: 
      chi_df.loc[i, var] = 4
    elif chi_df.loc[i, var] <= 0.6: 
      chi_df.loc[i, var] = 6
    elif chi_df.loc[i, var] <= 0.8: 
      chi_df.loc[i, var] = 8
    else: 
      chi_df.loc[i, var] = 10
  chi_df[var] = chi_df[var].map({2: "0.0-0.2", 4: "0.2-0.4", 6: "0.4-0.6", 8: "0.6-0.8", 10: "0.8-1.0"})
  chi_df = chi_df.groupby(by=["party", var], as_index=False).sum()
  chi_df = chi_df.pivot(index=var, columns="party", values="count")
  chi_df = chi_df.fillna(0)
  chi2, p, dof, ex = scipy.stats.chi2_contingency(chi_df, correction=False)
  print("P-value: {}".format(p))
  exp_df = pd.DataFrame(ex)
  exp_df = exp_df.rename(columns={0: "DEM", 1: "LIB", 2: "OTH", 3: "REP"})
  exp_df = exp_df.set_index(chi_df.index)
  display(post_hoc(chi_df, exp_df))
  print()

# post hoc
obs_df = pivot_df
exp_df = pd.DataFrame(ex)
exp_df.set_index([["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]], inplace=True)
exp_df = exp_df.rename(columns={0: "DEM", 1: "LIB", 2: "OTH", 3: "REP"})
post_hoc(obs_df, exp_df)

# chi-squared tests for incumbency
pivot_df = pd.concat([sen_data["CAND_ICI"], sen_data["candidatevotes"] / sen_data["totalvotes"]], axis=1)
pivot_df = pivot_df.rename({"CAND_ICI": "CAND_ICI", 0: "vote_percent"}, axis=1)
pivot_df["count"] = 1
for i in range(pivot_df.shape[0]): 
  if pivot_df.loc[i, "vote_percent"] <= 0.2: 
    pivot_df.loc[i, "vote_percent"] = 2
  elif pivot_df.loc[i, "vote_percent"] <= 0.4: 
    pivot_df.loc[i, "vote_percent"] = 4
  elif pivot_df.loc[i, "vote_percent"] <= 0.6: 
    pivot_df.loc[i, "vote_percent"] = 6
  elif pivot_df.loc[i, "vote_percent"] <= 0.8: 
    pivot_df.loc[i, "vote_percent"] = 8
  else: 
    pivot_df.loc[i, "vote_percent"] = 10
pivot_df["vote_percent"] = pivot_df["vote_percent"].map({2: "0.0-0.2", 4: "0.2-0.4", 6: "0.4-0.6", 8: "0.6-0.8", 10: "0.8-1.0"})
pivot_df = pivot_df.groupby(by=["CAND_ICI", "vote_percent"], as_index=False).sum()
pivot_df = pivot_df.pivot(index="vote_percent", columns="CAND_ICI", values="count")
pivot_df = pivot_df.fillna(0)
chi2, p, dof, ex = scipy.stats.chi2_contingency(pivot_df, correction=False)
print(p)

# post hoc
obs_df = pivot_df
exp_df = pd.DataFrame(ex)
exp_df.set_index([["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]], inplace=True)
exp_df = exp_df.rename(columns={0: "C", 1: "I", 2: "O"})
post_hoc(obs_df, exp_df)

for var in ["indiv_percent", "pol_pty_percent", "other_pac_percent"]: 
  chi_df = pd.concat([sen_data["CAND_ICI"], sen_data[var]], axis=1)
  chi_df["count"] = 1
  if var == "indiv_percent": 
    print("Percent of Contributions from Individuals")
  elif var == "pol_pty_percent": 
    print("Percent of Contributions from Political Parties")
  else: 
    print("Percent of Contributions from PACs")
  for i in range(chi_df.shape[0]): 
    if chi_df.loc[i, var] <= 0.2: 
      chi_df.loc[i, var] = 2
    elif chi_df.loc[i, var] <= 0.4: 
      chi_df.loc[i, var] = 4
    elif chi_df.loc[i, var] <= 0.6: 
      chi_df.loc[i, var] = 6
    elif chi_df.loc[i, var] <= 0.8: 
      chi_df.loc[i, var] = 8
    else: 
      chi_df.loc[i, var] = 10
  chi_df[var] = chi_df[var].map({2: "0.0-0.2", 4: "0.2-0.4", 6: "0.4-0.6", 8: "0.6-0.8", 10: "0.8-1.0"})
  chi_df = chi_df.groupby(by=["CAND_ICI", var], as_index=False).sum()
  chi_df = chi_df.pivot(index=var, columns="CAND_ICI", values="count")
  chi_df = chi_df.fillna(0)
  chi2, p, dof, ex = scipy.stats.chi2_contingency(chi_df, correction=False)
  print("P-value: {}".format(p))
  exp_df = pd.DataFrame(ex)
  exp_df = exp_df.rename(columns={0: "C", 1: "I", 2: "O"})
  exp_df = exp_df.set_index(chi_df.index)
  display(post_hoc(chi_df, exp_df))
  print()

