# SSI-2021
Code for my project from Summer STEM Institute, 2021: **Analyzing the Relationships Between PAC Contributions, Incumbency, Political Party, and Campaign Success**

## Project Description
Election finance affects all political campaigns and defines those who win political power and influence. Understanding this relationship can highlight the influence of money in politics and the ways in which it affects democracy. This project examines how reliance on different sources of funding — individuals, political parties, and political action committees (PACs) — affect the number of votes won by a political campaign. It will also study factors already known to affect election chances and how they interact with campaign success. In this project, linear regression models were used to answer the question of how sources of funding affect the percentage of votes won by a political candidate. Chi-Squared tests were used to analyze whether the categorical factors of incumbency and political party had statistically significant relationships with votes won or financial data. It was found that PAC money had a stronger relationship with votes won than individual contributions. Also, challengers are more likely to rely heavily on individual donations, while incumbents are more likely to rely on PAC money. Meanwhile, incumbents and members of the Democratic and Republican party were most likely to rely heavily on PAC funding. Policymakers who wish to combat this influence may use these results to avoid unwarranted financial influence in government.

## Skills I Learned
 - Merging datasets (even with discrepancies among columns, e.g. names being spelled differently, middle initials included in one dataset and not the other)
 - Manipulating `pandas` dataframes
 - Making heatmaps with data to reduce multicollinearity
 - Fitting a linear regression for maximum accuracy  (including how to pick the best variables)
 - How to use the libraries `statsmodels`, `scipy.stats`, `sklearn.metrics` for machine learning in Python
 - Basic R programming for linear regression
 - Chi-squared tests in Python

## Datasets Used
Federal Election Commission, 2021, “All Candidates”, https://www.fec.gov/data/browse-data/?tab=bulk-data#bulk-data

MIT Election Data and Science Lab, 2017, "U.S. House 1976–2018", https://doi.org/10.7910/DVN/IG0UN2, Harvard Dataverse, V8, UNF:6:p05gglERZ/Fe5LP4RarxeA== [fileUNF]

MIT Election Data and Science Lab, 2017, "U.S. President 1976–2020", https://doi.org/10.7910/DVN/42MVDX, Harvard Dataverse, V6, UNF:6:4KoNz9KgTkXy0ZBxJ9ZkOw== [fileUNF]

MIT Election Data and Science Lab, 2017, "U.S. Senate 1976–2020", https://doi.org/10.7910/DVN/PEJ5QU, Harvard Dataverse, V5, UNF:6:cIUB3CEIKhMi9tiY4BffLg== [fileUNF]

