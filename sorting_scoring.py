#############################################
# RATING PRODUCT & SORTING REVIEWS in AMAZON
#############################################

#####################
# Data Understanding
#####################

# Necessary libraries are imported.
import numpy as np
import pandas as pd
import math
import scipy.stats as st

from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


df_ = pd.read_csv("amazon_review.csv")
df = df_.copy()


#####################
# Data Overview
#####################

def check_df(df, head=5):
    print("-------------------- Shape --------------------")
    print(df.shape)
    print("\n-------------------- Types --------------------")
    print(df.dtypes)
    print("\n-------------------- Head --------------------")
    print(df.head(head))
    print("\n-------------------- Tail --------------------")
    print(df.tail(head))
    print("\n-------------------- Na --------------------")
    print(df.isnull().sum())
    print("\n-------------------- Quantiles --------------------")
    print(df.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print("\n-------------------- Features Names --------------------")
    print(df.columns)

check_df(df)


df.info()

df.nunique()


#####################
#Product Rating
#####################

# Since it is a single product, I directly take the average and observe it.
df["overall"].mean()

# Calculate the weighted average score by date.
df["reviewTime"] = pd.to_datetime(df["reviewTime"])

current_date = df["reviewTime"].max()

df["day_diff"] = (current_date - df["reviewTime"]).dt.days

df.head()


df["day_diff"].quantile([.25, .5, .75])

# Weight calculation with quantile function
print(df.loc[(df["day_diff"] <= 280), "overall"].mean())
print(df.loc[(df["day_diff"] > 280) & (df["day_diff"] <= 430), "overall"].mean())
print(df.loc[(df["day_diff"] > 430) & (df["day_diff"] <= 600), "overall"].mean())
print(df.loc[(df["day_diff"] > 600), "overall"].mean())


# writing with function
def time_based_weighted_average(dataframe, w1=30, w2=28, w3=24, w4=18):
    return dataframe.loc[dataframe["day_diff"] <= 280, "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 280) & (dataframe["day_diff"] <=430), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 430) & (dataframe["day_diff"] <=600), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 600), "overall"].mean() * w4 / 100

time_based_weighted_average(df)


time_based_weighted_average(df, 30, 26, 22, 22)

#how many people were found in each scoring range.
df.groupby("asin").agg({"overall" : "value_counts"})


#####################
# Product Reviews
#####################

# The number of those who did not find the comment useful was calculated.
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]
df.head()


df["helpful_no"].value_counts()


# score_pos_nef_diff
# Scoring according to the difference between up and down
def score_up_down_diff(up, down):
    return up - down

df["score_pos_neg_diff"] = score_up_down_diff(df["helpful_yes"], df["helpful_no"])

df.sort_values(by = "score_pos_neg_diff", ascending = False).head()


# score_average_rating
# It scores the comment by dividing the number of ups by the total number of votes.
def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"],x["helpful_no"]), axis=1 )

df.sort_values("score_average_rating", ascending = False).head()


# wilson_lower_bound

"""
     Calculate Wilson Lower Bound Score

     - The lower limit of the confidence interval to be calculated for the Bernoulli parameter p is accepted as the WLB score.
     - The score to be calculated is used for product ranking.
     - Note:
     If the scores are between 1-5, 1-3 are marked as negative, 4-5 as positive and can be made to conform to Bernoulli.
     This brings with it some problems. For this reason, it is necessary to make a bayesian average rating.

     parameters
     ----------
     up: int
         up count
     down: int
         down count
     confidence: float
         confidence

     Returns
     -------
     wilson score: float

     """

def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)


wilson_lower_bound(600, 400)

wilson_lower_bound(5500, 4500)

df.sort_values("wilson_lower_bound", ascending=False).head(20)


# As a result, when we look at the table, the comments in the lower ranks were at the top of the list with the high wilson_lower_bound values.
# The fact that even the comments given 1 star find themselves in the top ranks is also due to this lower-upper ranking.