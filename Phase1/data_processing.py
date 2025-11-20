import pandas as pd
import yfinance as yf

# Example: Tesla stock
stock_prices = yf.download("TSLA", start="2015-01-01", end="2019-12-31")
#calculate the return of the stock
stock_prices["Return"] = (stock_prices["Close"]-stock_prices["Open"])/stock_prices["Open"]
#reset the index and get the columns
stock_prices = stock_prices.reset_index()
stock_prices.columns = stock_prices.columns.get_level_values(0)
stock_prices.to_csv("data/TSLA_2015_2020.csv")


stock_prices=pd.read_csv("data/TSLA_2015_2020.csv")
tweets = pd.read_csv("data/tsla_dataset.csv")

#convert the post_date to datetime
tweets["post_date"] = pd.to_datetime(tweets["post_date"])
stock_prices["Date"] = pd.to_datetime(stock_prices["Date"])

#extract the date only
tweets["date"] = tweets["post_date"].dt.date
stock_prices["date"] = stock_prices["Date"].dt.date

#get the trading dates
trading_dates = stock_prices["Date"].dt.date

#get the next trading day for each tweet
def get_next_trading_day(tweet_date):
    later = trading_dates[trading_dates >= tweet_date]
    return later.min() if not later.empty else None

tweets["effective_date"] = tweets["post_date"].dt.date.apply(get_next_trading_day)


tweets["effective_date"] = pd.to_datetime(tweets["effective_date"])

#merge the stock prices and the tweets
data=pd.merge(stock_prices, tweets,left_on="Date",right_on="effective_date", how="inner")

data.drop(columns=["Date"], inplace=True)
data.to_csv("data/TSLA_2015_2020_with_tweets.csv")

start_date = "2015-01-01"
end_date = "2016-01-01"

filtered = data[(data["post_date"] >= start_date) & (data["post_date"] <= end_date)]

filtered.to_csv("data/eda_data.csv", index=False)


