# R has the functionality for a linear regression that chooses the best subsets of variables to use, factoring in multicollinearity and the accuracy of the model.
# I used R to make my linear regression model more accuracte.
# This was done for both House and Senate data.

library(leaps)
library(readr)
library(tidyverse)
sen_data <- read_csv("~/Desktop/SSI 2021 Project Folder/sen_data.csv")

vars = c("TTL_RECEIPTS", "TRANS_FROM_AUTH", "TTL_DISB", "TRANS_TO_AUTH",
         "COH_BOP", "COH_COP", "CAND_CONTRIB", "CAND_LOANS", "OTHER_LOANS",
         "CAND_LOAN_REPAY", "OTHER_LOAN_REPAY", "DEBTS_OWED_BY",
         "TTL_INDIV_CONTRIB", "OTHER_POL_CMTE_CONTRIB", "POL_PTY_CONTRIB",
         "INDIV_REFUNDS", "CMTE_REFUNDS","total_contrib", "indiv_percent",
         "pol_pty_percent", "other_pac_percent", "candidatevotes", "totalvotes", "vote_percent")
predictors = c("TTL_RECEIPTS", "TRANS_FROM_AUTH", "TTL_DISB", "TRANS_TO_AUTH",
               "COH_BOP", "COH_COP", "CAND_CONTRIB", "CAND_LOANS", "OTHER_LOANS",
               "CAND_LOAN_REPAY", "OTHER_LOAN_REPAY", "DEBTS_OWED_BY",
               "TTL_INDIV_CONTRIB", "OTHER_POL_CMTE_CONTRIB", "POL_PTY_CONTRIB",
               "INDIV_REFUNDS", "CMTE_REFUNDS","total_contrib", "indiv_percent",
               "pol_pty_percent", "other_pac_percent")

split_ind <- as.integer(dim(sen_data)[1] * 0.8)
train <- slice(sen_data, 1:split_ind)
train <- train[vars]
train[,"vote_percent"] = train[,"candidatevotes"] / train[,"totalvotes"]
train <- na.omit(train)
test <- slice(sen_data, split_ind:dim(sen_data)[1])
test <- test[vars]
test[,"vote_percent"] = test[,"candidatevotes"] / test[,"totalvotes"]
test <- na.omit(test)

best_subsets <- regsubsets(y=as.matrix(train[c("vote_percent")]), x=as.matrix(train[predictors]),
                           # TTL_RECEIPTS+TRANS_FROM_AUTH+TTL_DISB+
                           # TRANS_TO_AUTH+COH_BOP+COH_COP+CAND_CONTRIB+
                           # CAND_LOANS+OTHER_LOANS+CAND_LOAN_REPAY+
                           # OTHER_LOAN_REPAY+DEBTS_OWED_BY+TTL_INDIV_CONTRIB+
                           # OTHER_POL_CMTE_CONTRIB+POL_PTY_CONTRIB+INDIV_REFUNDS+
                           # CMTE_REFUNDS+total_contrib+indiv_percent+
                           # pol_pty_percent+other_pac_percent, data = train,
                           n_best = length(predictors), method = "exhaustive", really.big = TRUE)
summ <- summary(best_subsets)
subsets <- cbind(summ$outmat, summ$adjr2)
subsets <- subsets[order(as.numeric(subsets[,ncol(subsets)])),]
