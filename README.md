📈 NIFTY 50 Stock Analyzer — KMeans + PCA + Decision Tree
A multi-stage unsupervised and supervised learning pipeline that clusters NIFTY 50 stocks by price behavior, evaluates cluster profitability, and builds a decision tree to classify new stocks — helping identify stocks worth investing in based on historical Open, High, and Low prices.

📌 Problem Statement
With 50 stocks on the NIFTY index behaving differently across market conditions, can we group stocks by their price structure and identify which groups have historically been profitable? This project uses clustering to discover natural stock groupings, evaluates each group's profitability, and trains a classifier to predict which cluster a new stock belongs to — enabling a data-driven investment signal.

📂 Dataset

Source: NIFTY 50 Stocks Dataset — NSE India
Exchange: National Stock Exchange of India (NSE)
Features used: Open, High, Low, LTP, Chng, % Chng, 30 d % chng, 365 d % chng, 52w H, 52w L, Volume (lacs)
Preprocessing note: Comma-separated numeric strings cleaned and converted to float before modeling


🔍 Exploratory Data Analysis
15+ visualizations were created to understand price dynamics and return relationships:

Open vs 365d % change (lineplot): Long-term return patterns across price ranges
High/Low vs 30d & 365d % change (lineplot): Short and long-term momentum by price level
Open vs Chng / % Chng (barplot): Daily change distribution across opening prices
Chng vs 52w High/Low (scatterplot): Relationship between daily movement and 52-week extremes
LTP vs 30d & 365d % change (lineplot): Last traded price momentum analysis
Volume vs Chng (jointplot KDE): Trading volume and daily price change joint distribution


🏗️ Pipeline Architecture
Raw NIFTY 50 Data
      │
      ▼
  Data Cleaning
  (remove commas, cast to numeric)
      │
      ▼
  KMeans Clustering (n=4)
  Features: Open, High, Low
      │
      ▼
  PCA Visualization (2D)
  Confirm cluster separation
      │
      ▼
  Profitability Evaluation
  profit = LTP - Open
  cluster_profitability = % of profitable days per cluster
      │
      ▼
  Investment Signal
  threshold = 0.3 → Worth Investing / Not Worth Investing
      │
      ▼
  Decision Tree Classifier (max_depth=3)
  Learns cluster boundaries from Open, High, Low
  Enables fast cluster prediction for new stocks

🤖 Models
1. KMeans Clustering

Features: Open, High, Low
Clusters: 4 (n_clusters=4, random_state=0)
Purpose: Group stocks by price structure without labels

2. PCA (Principal Component Analysis)

Components: 2
Purpose: Reduce 3D cluster space to 2D for visualization — confirms clusters are meaningfully separated

3. Decision Tree Classifier

Features: Open, High, Low
Target: KMeans cluster label
Max depth: 3
Purpose: Approximate the KMeans decision boundary with an interpretable rule-based classifier — allows fast, explainable cluster prediction on new data


📊 Results
Cluster profitability (proportion of records where LTP > Open):
Cluster         Profitable               %InvestmentSignal
0               Available in notebook    Threshold > 0.3 → Worth Investing
1               Available in notebook    Threshold > 0.3 → Worth Investing
2               Available in notebook    Threshold ≤ 0.3 → Not Worth Investing
3               Available in notebook    Threshold ≤ 0.3 → Not Worth Investing

Example prediction:
Input: Open=344.05, High=346.50, Low=338.0
→ KMeans assigns cluster
→ Profitability check against threshold
→ Output: "Worth Investing" / "Not Worth Investing"

💡 Key Design Decisions
Why KMeans on Open, High, Low?
These three features define a stock's intraday price range — the spread between them captures volatility and price level simultaneously. Clustering on these groups stocks with structurally similar trading behavior.
Why PCA for visualization?
With 3 clustering features, direct visualization isn't possible. PCA compresses to 2 dimensions while preserving maximum variance — confirming that the 4 clusters are genuinely separated and not arbitrary.
Why a Decision Tree after KMeans?
KMeans prediction requires loading the full model and computing distances to centroids. A shallow decision tree (depth=3) learns the same boundaries as a set of simple if-else rules — making the investment logic transparent, fast, and explainable to a non-technical user.
Why a profitability threshold of 0.3?
A cluster is flagged as investable if more than 30% of its records showed LTP > Open (i.e., the stock closed above its opening price). This is a conservative baseline signal — not a trading strategy.

⚠️ Known Limitations

n_clusters=4 is not formally justified — an elbow method or silhouette score analysis should be used to validate this choice
Circular reasoning risk: The Decision Tree is trained to predict KMeans labels from the same features used for clustering. It is not learning to predict profitability directly — it is approximating the KMeans boundary
Threshold of 0.3 is heuristic — not derived from backtesting or statistical analysis
Small dataset: NIFTY 50 contains only 50 stocks — clustering on 50 points is statistically weak
No temporal dimension: The dataset is a snapshot, not a time series — this is not a trading signal system
Not financial advice: This project is an ML learning exercise, not an investment tool


🛠️ Tech Stack

Python 3
Pandas, NumPy
Matplotlib, Seaborn
Scikit-learn (KMeans, PCA, DecisionTreeClassifier)
Graphviz (Decision Tree visualization)
