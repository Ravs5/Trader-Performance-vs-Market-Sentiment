Trader Performance vs Market Sentiment

Project Overview



This project analyzes how market sentiment (Fear vs Greed) influences trader performance, including profitability, win rate, volatility, trading behavior, and position sizing.



Daily Fear \& Greed Index data was merged with historical trade-level performance data to uncover behavioral patterns under different market conditions and to build a predictive model for daily profitability.



Project Structure



Trader-Performance-vs-Market-Sentiment/



Trader\_Performance\_vs\_Market\_Sentiment.ipynb



Trader\_Performance\_vs\_Market\_Sentiment.py



Trader\_Performance\_vs\_Market\_Sentiment.html



fear\_greed\_index.csv



historical\_data.csv



README.md



requirements.txt



Setup Instructions



Clone the repository:



git clone https://github.com/Ravs5/Trader-Performance-vs-Market-Sentiment.git



cd Trader-Performance-vs-Market-Sentiment



Install dependencies:



pip install -r requirements.txt



It is recommended to use a virtual environment.



How to Run



Option 1: Run the Notebook (Recommended)



Open Trader\_Performance\_vs\_Market\_Sentiment.ipynb and run all cells sequentially to reproduce the full analysis, visualizations, and modeling results.



Option 2: Run the Python Script



python Trader\_Performance\_vs\_Market\_Sentiment.py



This will execute the complete data cleaning, merging, exploratory analysis, and modeling pipeline.



1\. Objective



This project analyzes how market sentiment (Fear vs Greed) influences trader performance, including profitability, win rate, volatility, and trading behavior.



We merged daily Fear \& Greed Index data with historical trade-level performance to uncover behavioral patterns under different market conditions.



2\. Methodology

Data Preparation



Converted date columns to datetime format



Merged sentiment and trade data on aligned dates



Created derived features:



win (PnL > 0)



Daily aggregated PnL



profitable\_day (binary target)



Sentiment encoding (Fear = 0, Greed = 1)



Exploratory Analysis



Compared:



Average PnL by sentiment



Win rate by sentiment



PnL volatility



Trade frequency



Position size behavior



Segmented traders into:



Frequent vs Infrequent



Consistent vs Inconsistent



Predictive Modeling



Aggregated daily trader-level data



Built a Logistic Regression model



Features used:



Sentiment



Average position size



Win ratio



3\. Model Performance



Accuracy: 93.89%



Classification Results:



Profitable Day Precision: 0.97



Profitable Day Recall: 0.94



Losing Day Precision: 0.89



Losing Day Recall: 0.94



The model demonstrates strong predictive capability in identifying both profitable and non-profitable trading days.



4\. Key Insights



Traders perform more consistently during Greed phases.



Extreme Fear increases volatility and performance instability.



Win rate declines during panic-driven markets.



Position sizes vary with sentiment, suggesting emotional risk behavior.



Frequent traders are more sensitive to sentiment swings.



Consistent traders maintain lower performance variance across conditions.



5\. Strategic Recommendations



Reduce position size during Extreme Fear periods.



Apply stricter risk controls in volatile sentiment phases.



Integrate sentiment signals into algorithmic trading models.



Encourage rule-based trading to reduce emotional overtrading.



Use trader segmentation for personalized risk management frameworks.



6\. Conclusion



Market sentiment significantly impacts trading behavior, risk exposure, and return stability.



While sentiment does not directly determine profitability, it meaningfully influences volatility, consistency, and trader decision-making patterns.



Incorporating sentiment into trading strategies can improve robustness and reduce emotionally driven losses.

