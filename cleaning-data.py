# For this project, I needed to combine data from the FEC dataset (campaign finance) and MEDSL datasets (election results). Because of small discrepancies 
# in how candidates' names were entered into the dataset, I used both the pandas library in python and Google Sheets functionality to combine the datasets.
# The Google Sheets functionality is difficult to document, but a quick explanation would be that I split the candidates' names into three columns: first name,
# last name, and middle initial. The rest is done in the following python code.
# Note: The cleaned datasets used in finding-relationships.py are also uploaded into this repository for ease of access.

# importing libraries
from google.colab import auth
import gspread
from oauth2client.client import GoogleCredentials
import pandas as pd

# converts the MEDSL-formatted names to "first last" once suffixes have been separated
def mit_convert(name): 
  first = name[:name.find(" ")]
  last = ""
  for i in range(-1, -1 * len(name) - 1, -1): 
    if name[i] == " ": 
      last = name[i + 1:]
      break
  return "{} {}".format(first, last)

# loading data from Google Sheets

# authenticating Google Sheets
auth.authenticate_user()
gc = gspread.authorize(GoogleCredentials.get_application_default())

# making a list of all the years included in the data
years = []
for i in range(1980, 2022, 2): # years does not include the year 2022, because range is exclusive for the second value
  years.append(str(i))

# concatting the data from sheets
fec_header = ["CAND_ID", "CAND_ICI", "PTY_CD", "CAND_PTY_AFFILIATION", "TTL_RECEIPTS", "TRANS_FROM_AUTH", "TTL_DISB", "TRANS_TO_AUTH", "COH_BOP",
                             "COH_COP", "CAND_CONTRIB", "CAND_LOANS", "OTHER_LOANS", "CAND_LOAN_REPAY", "OTHER_LOAN_REPAY", "DEBTS_OWED_BY", "TTL_INDIV_CONTRIB", "CAND_OFFICE_ST",
                             "CAND_OFFICE_DISTRICT", "OTHER_POL_CMTE_CONTRIB", "POL_PTY_CONTRIB", "CVG_END_DT", "INDIV_REFUNDS", "CMTE_REFUNDS", "last_name", "suffix", "first_name"]
fec_data = pd.DataFrame(columns=fec_header)
for i in years: 
  temp = pd.DataFrame(gc.open_by_url("URL omitted for account privacy (see data folder for cleaned data)").worksheet(i).get_all_values(), columns=fec_header)
  temp["year"] = i
  fec_data = pd.concat([temp[1:], fec_data])
fec_data = fec_data.reset_index(drop=True)

# MIT data
mit_sen = pd.DataFrame(gc.open_by_url("URL omitted for account privacy (see data folder for cleaned data)").worksheet("Senate").get_all_values())
mit_sen.columns = mit_sen.iloc[0]
mit_sen = mit_sen[1:]
mit_sen["candidate"] = mit_sen["candidate"].apply(mit_convert)
temp = mit_sen["candidate"].str.split(" ", expand=True)
mit_sen["first_name"] = temp[0]
mit_sen["last_name"] = temp[1]

mit_house = pd.DataFrame(gc.open_by_url("URL omitted for account privacy (see data folder for cleaned data)").worksheet("House").get_all_values())
mit_house.columns = mit_house.iloc[0]
mit_house = mit_house[1:]
mit_house["candidate"] = mit_house["candidate"].apply(mit_convert)
temp = mit_house["candidate"].str.split(" ", expand=True)
mit_house["first_name"] = temp[0]
mit_house["last_name"] = temp[1]

fec_data["CAND_OFFICE_DISTRICT"] = fec_data["CAND_OFFICE_DISTRICT"].replace({"": "00"})
fec_data = fec_data.rename(columns={"CAND_PTY_AFFILIATION": "party", "CAND_OFFICE_ST": "state_po", "CAND_OFFICE_DISTRICT": "district"})

# clean house data by type of election
fec_house = fec_data[fec_data["CAND_ID"].str[0] == "H"]
fec_house = fec_house.reset_index(drop=True)
print("Before: {}".format(mit_house.shape[0]))

mit_house["party"] = mit_house["party"].map({"REPUBLICAN": "REP", "DEMOCRAT": "DEM", "LIBERTARIAN": "LIB", "INDEPENDENT": "IND", "CONSTITUION": "CON"})
mit_house["party"] = mit_house["party"].fillna("OTH")

fec_house["party"] = fec_house["party"].map({"REP": "REP", "GOP": "REP", "DEM": "DEM", "LIB": "LIB", "IND": "IND", "CON": "CON"})
fec_house["party"] = fec_house["party"].fillna("OTH")

fec_house["district"] = fec_house["district"].replace({"": 0})
fec_house["district"] = fec_house["district"].dropna()
fec_house[["district", "year"]] = fec_house[["district", "year"]].astype(int)
mit_house[["district", "year"]] = mit_house[["district", "year"]].astype(int)

house_data = fec_house.merge(mit_house, on=["state_po", "year", "party", "district", "last_name"])
print("After: {}".format(house_data.shape[0]))

# clean senate data by type of election
mit_sen = mit_sen.rename(columns={"party_simplified": "party"})
mit_sen["party"] = mit_sen["party"].map({"REPUBLICAN": "REP", "DEMOCRAT": "DEM", "LIBERTARIAN": "LIB", "INDEPENDENT": "IND", "CONSTITUION": "CON"})
mit_sen["party"] = mit_sen["party"].fillna("OTH")

fec_sen = fec_data[fec_data["CAND_ID"].str[0] == "S"]
fec_sen = fec_sen.reset_index(drop=True)
print("Before: {}".format(mit_sen.shape[0]))

fec_sen["party"] = fec_sen["party"].map({"REP": "REP", "GOP": "REP", "DEM": "DEM", "LIB": "LIB", "IND": "IND", "CON": "CON"})
fec_sen["party"] = fec_sen["party"].fillna("OTH")

sen_data = fec_sen.merge(mit_sen, on=["state_po", "year", "party", "last_name"])
print("After: {}".format(sen_data.shape[0]))

# calculations and final formatting
# some typecasting
house_data["TTL_INDIV_CONTRIB"] = house_data["TTL_INDIV_CONTRIB"].astype(float)
house_data["POL_PTY_CONTRIB"] = house_data["POL_PTY_CONTRIB"].astype(float)
house_data["OTHER_POL_CMTE_CONTRIB"] = house_data["OTHER_POL_CMTE_CONTRIB"].astype(float)
house_data["TTL_DISB"] = house_data["TTL_DISB"].astype(float)
house_data["candidatevotes"] = house_data["candidatevotes"].astype(float)
house_data["totalvotes"] = house_data["totalvotes"].astype(float)

sen_data["TTL_INDIV_CONTRIB"] = sen_data["TTL_INDIV_CONTRIB"].astype(float)
sen_data["POL_PTY_CONTRIB"] = sen_data["POL_PTY_CONTRIB"].astype(float)
sen_data["OTHER_POL_CMTE_CONTRIB"] = sen_data["OTHER_POL_CMTE_CONTRIB"].astype(float)
sen_data["TTL_DISB"] = sen_data["TTL_DISB"].astype(float)
sen_data["candidatevotes"] = sen_data["candidatevotes"].astype(float)
sen_data["totalvotes"] = sen_data["totalvotes"].astype(float)

# final calculations (percentage of each source of contributions, percentage of vote won, etc.)
house_data["total_contrib"] = house_data["TTL_INDIV_CONTRIB"] + house_data["POL_PTY_CONTRIB"] + house_data["OTHER_POL_CMTE_CONTRIB"]
house_data["indiv_percent"] = (house_data["TTL_INDIV_CONTRIB"] / house_data["total_contrib"]) * 100
house_data["pol_pty_percent"] = (house_data["POL_PTY_CONTRIB"] / house_data["total_contrib"]) * 100
house_data["other_pac_percent"] = (house_data["OTHER_POL_CMTE_CONTRIB"] / house_data["total_contrib"]) * 100
house_data["usd_per_vote"] = house_data["TTL_DISB"] / house_data["candidatevotes"]
house_data["usd_per_percent"] = house_data["TTL_DISB"] / ((house_data["candidatevotes"] / house_data["totalvotes"]) * 100)

sen_data["total_contrib"] = sen_data["TTL_INDIV_CONTRIB"] + sen_data["POL_PTY_CONTRIB"] + sen_data["OTHER_POL_CMTE_CONTRIB"]
sen_data["indiv_percent"] = (sen_data["TTL_INDIV_CONTRIB"] / sen_data["total_contrib"]) * 100
sen_data["pol_pty_percent"] = (sen_data["POL_PTY_CONTRIB"] / sen_data["total_contrib"]) * 100
sen_data["other_pac_percent"] = (sen_data["OTHER_POL_CMTE_CONTRIB"] / sen_data["total_contrib"]) * 100
sen_data["usd_per_vote"] = sen_data["TTL_DISB"] / sen_data["candidatevotes"]
sen_data["usd_per_percent"] = sen_data["TTL_DISB"] / ((sen_data["candidatevotes"] / sen_data["totalvotes"]) * 100)

# fixing some undescriptive column names
house_data = house_data.drop(columns=["suffix_x", "first_name_x"])
house_data = house_data.rename(columns={"suffix_y": "suffix", "first_name_y": "first_name"})

sen_data = sen_data.drop(columns=["suffix_x", "first_name_x"])
sen_data = sen_data.rename(columns={"suffix_y": "suffix", "first_name_y": "first_name"})

# downloading data to CSV
# downloading data
house_data.to_csv("house_data.csv")
sen_data.to_csv("sen_data.csv")
