# GrowthHub Retention Analytics Dashboard

An interactive Streamlit dashboard analyzing SaaS retention patterns, churn risk, and customer lifetime value.

## 🚀 Live Demo

[View Live Dashboard](#) *(Add your deployed URL here)*

## 📊 Features

- **Overview Dashboard**: Key metrics, MRR tracking, user growth
- **Cohort Analysis**: Interactive retention heatmaps and curves
- **Churn Risk Scoring**: Identify at-risk users and revenue
- **Customer Segmentation**: LTV analysis and value-based segments
- **Plan Comparison**: Retention metrics across subscription tiers

## 🛠️ Tech Stack

- **Python 3.8+**
- **Streamlit**: Interactive web dashboard
- **Plotly**: Interactive visualizations
- **Pandas & NumPy**: Data manipulation

## 📦 Installation

### Local Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd growthhub-dashboard
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run streamlit_app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## ☁️ Deploy to Streamlit Cloud (FREE)

### Step 1: Prepare Your Repository

1. Create a GitHub repository
2. Upload these files:
   - `streamlit_app.py`
   - `requirements.txt`
   - `README.md`

### Step 2: Deploy

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set main file: `streamlit_app.py`
6. Click "Deploy"!

Your app will be live at: `https://your-app-name.streamlit.app`

## 📈 Dashboard Pages

### 1. Overview
- Total users, active users, churn rate
- Monthly signup trends
- Plan distribution
- Revenue by plan tier
- Retention milestones

### 2. Cohort Analysis
- Monthly cohort retention heatmap
- Average retention curve
- Drill-down by plan and date range

### 3. Churn Risk
- High/medium/low risk segmentation
- MRR at risk
- Top at-risk customers list
- Downloadable CSV export

### 4. Customer Segments
- LTV distribution and statistics
- Whale/High/Medium/Low value segments
- Revenue contribution analysis

### 5. Plan Comparison
- Churn rate by plan tier
- Engagement metrics comparison
- Retention curves by plan
- Comprehensive plan metrics table

## 🎯 Use Cases

- **Product Teams**: Understand feature adoption and engagement patterns
- **Customer Success**: Identify at-risk accounts for proactive intervention
- **Executive Leadership**: Monitor key retention and revenue metrics
- **Data Analysis Portfolio**: Showcase end-to-end analytics project

## 📊 Data

This dashboard uses synthetic data that mimics realistic SaaS behavior patterns:
- 5,000 users across 24 months
- 4 subscription tiers (Free, Starter, Professional, Enterprise)
- Realistic churn patterns based on engagement metrics
- Feature adoption, team size, and usage data

## 🔧 Customization

To use your own data, replace the `generate_data()` function in `streamlit_app.py` with your data loading logic:

```python
@st.cache_data
def load_data():
    df = pd.read_csv('your_data.csv')
    # Add any preprocessing here
    return df
```

## 📝 Project Structure

```
growthhub-dashboard/
├── streamlit_app.py      # Main dashboard application
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🤝 Contributing

This is a portfolio project, but suggestions are welcome! Feel free to open an issue or submit a pull request.

## 📄 License

MIT License - feel free to use this for your own portfolio projects.

## 👤 Author

**Your Name**
- Portfolio: [your-portfolio-link]
- LinkedIn: [your-linkedin]
- GitHub: [@your-github]

---

Built with ❤️ using Streamlit and Python
