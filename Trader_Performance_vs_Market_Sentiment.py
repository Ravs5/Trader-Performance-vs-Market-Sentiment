#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


# In[2]:


sentiment = pd.read_csv("fear_greed_index.csv")
trades = pd.read_csv("historical_data.csv")

print("Sentiment Shape:", sentiment.shape)
print("Trades Shape:", trades.shape)



# In[3]:


sentiment.head()


# In[4]:


trades.head()


# In[5]:


print("Sentiment Missing Values:\n")
print(sentiment.isnull().sum())

print("\nTrades Missing Values:\n")
print(trades.isnull().sum())


# In[6]:


print("Sentiment Duplicates:", sentiment.duplicated().sum())
print("Trades Duplicates:", trades.duplicated().sum())


# In[7]:


sentiment['date'] = pd.to_datetime(sentiment['date'])

trades['Timestamp IST'] = pd.to_datetime(trades['Timestamp IST'], dayfirst=True)

trades['date'] = trades['Timestamp IST'].dt.date

print("Sentiment Date Range:")
print(sentiment['date'].min(), "to", sentiment['date'].max())

print("\nTrader Date Range:")
print(trades['date'].min(), "to", trades['date'].max())


# In[8]:


sentiment['classification'].value_counts()


# In[9]:


sentiment['sentiment'] = sentiment['classification'].replace({
    'Extreme Fear': 'Fear',
    'Fear': 'Fear',
    'Greed': 'Greed',
    'Extreme Greed': 'Greed'
})


# In[10]:


trades['date'] = pd.to_datetime(trades['date'])


# In[11]:


merged = trades.merge(
    sentiment[['date','sentiment']],
    on='date',
    how='inner'
)

print("Merged Shape:", merged.shape)
merged.head()


# In[12]:


merged['win'] = merged['Closed PnL'] > 0


# In[13]:


daily_pnl = merged.groupby(['Account','date'])['Closed PnL'].sum().reset_index()
daily_pnl.head()


# In[14]:


win_rate = merged.groupby('Account')['win'].mean().reset_index()
win_rate.rename(columns={'win':'win_rate'}, inplace=True)
win_rate.head()


# In[15]:


trades_per_day = merged.groupby(['date','sentiment']).size().reset_index(name='trade_count')
trades_per_day.head()


# In[16]:


pd.crosstab(
    merged['sentiment'],
    merged['Side'],
    normalize='index'
)


# In[17]:


avg_pnl = merged.groupby('sentiment')['Closed PnL'].mean()
print(avg_pnl)


# In[18]:


plt.figure(figsize=(8,5))
sns.boxplot(x='sentiment', y='Closed PnL', data=merged)
plt.title("PnL Distribution by Market Sentiment")
plt.show()


# In[19]:


win_by_sentiment = merged.groupby('sentiment')['win'].mean()
print(win_by_sentiment)


# In[20]:


win_by_sentiment.plot(kind='bar', figsize=(6,4))
plt.title("Win Rate by Sentiment")
plt.ylabel("Win Rate")
plt.show()


# In[21]:


volatility = merged.groupby('sentiment')['Closed PnL'].std()
print(volatility)


# In[22]:


trade_freq = merged.groupby('sentiment').size()
print(trade_freq)

trade_freq.plot(kind='bar', figsize=(6,4))
plt.title("Number of Trades by Sentiment")
plt.show()


# In[23]:


size_by_sentiment = merged.groupby('sentiment')['Size USD'].mean()
print(size_by_sentiment)


# In[24]:


size_by_sentiment.plot(kind='bar', figsize=(6,4))
plt.title("Average Position Size by Sentiment")
plt.show()


# In[25]:


long_short = pd.crosstab(
    merged['sentiment'],
    merged['Side'],
    normalize='index'
)

print(long_short)


# In[26]:


sns.heatmap(long_short, annot=True, cmap="Blues")
plt.title("Long/Short Ratio by Sentiment")
plt.show()


# In[27]:


trade_counts = merged.groupby('Account').size()
median_trades = trade_counts.median()

merged['trader_type'] = merged['Account'].map(
    lambda x: 'Frequent' if trade_counts[x] > median_trades else 'Infrequent'
)

segment_pnl = merged.groupby(['trader_type','sentiment'])['Closed PnL'].mean().unstack()
print(segment_pnl)


# In[28]:


segment_pnl.plot(kind='bar', figsize=(8,5))
plt.title("Average PnL by Trader Type and Sentiment")
plt.show()


# In[29]:


pnl_stats = merged.groupby('Account')['Closed PnL'].agg(['mean','std']).reset_index()

threshold = pnl_stats['std'].median()

pnl_stats['consistency'] = pnl_stats['std'].apply(
    lambda x: 'Consistent' if x < threshold else 'Inconsistent'
)

merged = merged.merge(pnl_stats[['Account','consistency']], on='Account')

consistency_pnl = merged.groupby(['consistency','sentiment'])['Closed PnL'].mean().unstack()
print(consistency_pnl)

consistency_pnl.plot(kind='bar', figsize=(8,5))
plt.title("Average PnL by Consistency and Sentiment")
plt.show()


# In[30]:


daily_data = merged.groupby(['Account','date','sentiment']).agg({
    'Closed PnL':'sum',
    'Size USD':'mean',
    'win':'mean'
}).reset_index()

daily_data.head()


# In[31]:


daily_data['profitable_day'] = (daily_data['Closed PnL'] > 0).astype(int)

daily_data['sentiment_encoded'] = daily_data['sentiment'].map({
    'Fear': 0,
    'Greed': 1
})

daily_data[['Closed PnL','profitable_day','sentiment','sentiment_encoded']].head()


# In[32]:


from sklearn.model_selection import train_test_split

features = ['sentiment_encoded', 'Size USD', 'win']

model_data = daily_data[features + ['profitable_day']].dropna()

X = model_data[features]
y = model_data['profitable_day']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# In[33]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Model trained successfully.")


# In[34]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# In[38]:


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=['Not Profitable','Profitable'],
            yticklabels=['Not Profitable','Profitable'])
plt.title("Confusion Matrix")
plt.show()


# In[36]:


importance = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_[0]
})

importance = importance.sort_values(by='Coefficient', ascending=False)

importance


# In[37]:


trader_features = merged.groupby('Account').agg({
    'Closed PnL':'mean',
    'Size USD':'mean',
    'win':'mean'
}).reset_index()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_features = scaler.fit_transform(
    trader_features[['Closed PnL','Size USD','win']]
)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
trader_features['cluster'] = kmeans.fit_predict(scaled_features)

trader_features.head()

plt.figure(figsize=(8,5))
sns.scatterplot(
    x='Size USD',
    y='Closed PnL',
    hue='cluster',
    data=trader_features,
    palette='Set1'
)
plt.title("Trader Clusters")
plt.show()

