import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(
    page_title="GrowthHub Retention Analytics",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Generate data function
@st.cache_data
def generate_data():
    """Generate synthetic SaaS retention data"""
    np.random.seed(42)
    
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    num_users = 5000
    
    # Signup dates
    days_range = (end_date - start_date).days
    signup_days = np.random.beta(2, 5, num_users) * days_range
    signup_dates = [start_date + timedelta(days=int(d)) for d in signup_days]
    
    df = pd.DataFrame({
        'user_id': [f'user_{i:05d}' for i in range(num_users)],
        'signup_date': signup_dates
    })
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    df = df.sort_values('signup_date').reset_index(drop=True)
    
    # Plans
    plan_dist = {'Free': 0.70, 'Starter': 0.20, 'Professional': 0.08, 'Enterprise': 0.02}
    df['plan'] = np.random.choice(list(plan_dist.keys()), size=num_users, p=list(plan_dist.values()))
    
    plan_prices = {'Free': 0, 'Starter': 45, 'Professional': 800, 'Enterprise': 3200}
    df['mrr'] = df['plan'].map(plan_prices)
    
    # Engagement metrics
    def get_logins(plan):
        patterns = {'Free': (2, 5), 'Starter': (8, 4), 'Professional': (20, 6), 'Enterprise': (35, 8)}
        mean, std = patterns[plan]
        return max(0, np.random.normal(mean, std))
    
    df['monthly_logins'] = df['plan'].apply(get_logins)
    
    def get_team_size(plan):
        if plan == 'Free':
            return np.random.choice([1, 2, 3], p=[0.8, 0.15, 0.05])
        elif plan == 'Starter':
            return np.random.choice([1, 2, 3, 4, 5], p=[0.3, 0.3, 0.2, 0.1, 0.1])
        elif plan == 'Professional':
            return np.random.randint(3, 15)
        else:
            return np.random.randint(10, 50)
    
    df['team_size'] = df['plan'].apply(get_team_size)
    df['feature_adoption'] = np.random.beta(2, 3, num_users)
    
    contact_base = {'Free': 200, 'Starter': 800, 'Professional': 5000, 'Enterprise': 20000}
    df['contacts'] = df.apply(
        lambda x: int(contact_base[x['plan']] * x['feature_adoption'] * np.random.uniform(0.5, 1.5)),
        axis=1
    )
    
    df['campaigns_sent'] = np.random.poisson(df['monthly_logins'] * 0.3)
    df.loc[df['plan'] == 'Free', 'campaigns_sent'] = 0
    
    # Churn logic
    def calculate_churn_prob(row):
        base_rates = {'Free': 0.60, 'Starter': 0.35, 'Professional': 0.15, 'Enterprise': 0.08}
        prob = base_rates[row['plan']]
        
        if row['monthly_logins'] < 2:
            prob *= 1.5
        elif row['monthly_logins'] > 15:
            prob *= 0.5
        
        if row['team_size'] > 5:
            prob *= 0.6
        
        if row['feature_adoption'] > 0.6:
            prob *= 0.7
        elif row['feature_adoption'] < 0.2:
            prob *= 1.3
        
        if row['campaigns_sent'] > 5:
            prob *= 0.8
        
        return min(prob, 0.95)
    
    df['churn_prob'] = df.apply(calculate_churn_prob, axis=1)
    
    days_active = (end_date - df['signup_date']).dt.days
    can_churn = days_active > 30
    
    df['churned'] = False
    df.loc[can_churn, 'churned'] = np.random.random(can_churn.sum()) < df.loc[can_churn, 'churn_prob']
    
    def get_churn_date(row):
        if not row['churned']:
            return pd.NaT
        max_days = (end_date - row['signup_date']).days
        churn_day = min(int(np.random.exponential(90)), max_days - 1)
        return row['signup_date'] + timedelta(days=churn_day)
    
    df['churn_date'] = df.apply(get_churn_date, axis=1)
    
    df['days_active'] = df.apply(
        lambda x: (x['churn_date'] if pd.notna(x['churn_date']) else end_date) - x['signup_date'],
        axis=1
    ).dt.days
    
    df['ltv'] = df['mrr'] * (df['days_active'] / 30)
    df['cohort'] = df['signup_date'].dt.to_period('M')
    
    # Risk scoring for active users
    def risk_score(row):
        if row['churned']:
            return 0
        
        score = 0
        if row['monthly_logins'] < 5:
            score += 30
        elif row['monthly_logins'] < 10:
            score += 15
        
        if row['feature_adoption'] < 0.3:
            score += 25
        elif row['feature_adoption'] < 0.5:
            score += 12
        
        if row['team_size'] == 1:
            score += 20
        elif row['team_size'] <= 3:
            score += 10
        
        if row['campaigns_sent'] == 0:
            score += 15
        elif row['campaigns_sent'] < 3:
            score += 8
        
        thresholds = {'Free': 100, 'Starter': 500, 'Professional': 2000, 'Enterprise': 5000}
        if row['contacts'] < thresholds[row['plan']]:
            score += 10
        
        return min(score, 100)
    
    df['risk_score'] = df.apply(risk_score, axis=1)
    
    def risk_bucket(score):
        if score >= 70:
            return 'High'
        elif score >= 40:
            return 'Medium'
        return 'Low'
    
    df['risk_level'] = df['risk_score'].apply(risk_bucket)
    
    return df

# Load data
df = generate_data()

# Sidebar
st.sidebar.title("🎛️ Filters")

# Plan filter
plans = ['All'] + sorted(df['plan'].unique().tolist())
selected_plan = st.sidebar.selectbox("Plan", plans)

# Date range filter
min_date = df['signup_date'].min().date()
max_date = df['signup_date'].max().date()
date_range = st.sidebar.date_input(
    "Signup Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Apply filters
filtered_df = df.copy()
if selected_plan != 'All':
    filtered_df = filtered_df[filtered_df['plan'] == selected_plan]

if len(date_range) == 2:
    filtered_df = filtered_df[
        (filtered_df['signup_date'].dt.date >= date_range[0]) &
        (filtered_df['signup_date'].dt.date <= date_range[1])
    ]

# Navigation
page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "🔄 Cohort Analysis", "⚠️ Churn Risk", "💎 Customer Segments", "📈 Plan Comparison"]
)

# Main content
if page == "📊 Overview":
    st.markdown('<p class="main-header">GrowthHub Retention Analytics Dashboard</p>', unsafe_allow_html=True)
    st.markdown("**Real-time insights into user retention, churn risk, and customer lifetime value**")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_users = len(filtered_df)
        st.metric("Total Users", f"{total_users:,}")
    
    with col2:
        active_users = (~filtered_df['churned']).sum()
        active_pct = (active_users / total_users * 100) if total_users > 0 else 0
        st.metric("Active Users", f"{active_users:,}", f"{active_pct:.1f}%")
    
    with col3:
        churned_users = filtered_df['churned'].sum()
        churn_rate = (churned_users / total_users * 100) if total_users > 0 else 0
        st.metric("Churned Users", f"{churned_users:,}", f"{churn_rate:.1f}%")
    
    with col4:
        current_mrr = filtered_df[~filtered_df['churned']]['mrr'].sum()
        st.metric("Current MRR", f"${current_mrr:,.0f}")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Monthly Signups")
        monthly = filtered_df.groupby(filtered_df['signup_date'].dt.to_period('M')).size().reset_index()
        monthly.columns = ['month', 'signups']
        monthly['month'] = monthly['month'].astype(str)
        
        fig = px.bar(monthly, x='month', y='signups', 
                     labels={'month': 'Month', 'signups': 'Signups'},
                     color_discrete_sequence=['#1f77b4'])
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Plan Distribution")
        plan_counts = filtered_df['plan'].value_counts()
        
        fig = px.pie(values=plan_counts.values, names=plan_counts.index,
                     color_discrete_sequence=['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728'])
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Revenue by Plan")
        active = filtered_df[~filtered_df['churned']]
        mrr_by_plan = active.groupby('plan')['mrr'].sum().sort_values()
        
        fig = px.bar(x=mrr_by_plan.values, y=mrr_by_plan.index, orientation='h',
                     labels={'x': 'MRR ($)', 'y': 'Plan'},
                     color_discrete_sequence=['#1f77b4'])
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Retention Milestones")
        milestones = {
            '1 Month': (filtered_df['days_active'] >= 30).mean() * 100,
            '3 Months': (filtered_df['days_active'] >= 90).mean() * 100,
            '6 Months': (filtered_df['days_active'] >= 180).mean() * 100,
            '12 Months': (filtered_df['days_active'] >= 365).mean() * 100
        }
        
        fig = go.Figure(data=[
            go.Bar(x=list(milestones.values()), y=list(milestones.keys()), orientation='h',
                   marker_color='#2ca02c')
        ])
        fig.update_layout(xaxis_title="Retention Rate (%)", yaxis_title="", height=300)
        st.plotly_chart(fig, use_container_width=True)

elif page == "🔄 Cohort Analysis":
    st.markdown('<p class="main-header">Cohort Retention Analysis</p>', unsafe_allow_html=True)
    
    # Build retention table
    def build_retention_table(data, months=12):
        end_date = datetime(2024, 12, 31)
        data = data.copy()
        data['months_tenure'] = ((end_date.year - data['signup_date'].dt.year) * 12 + 
                                  (end_date.month - data['signup_date'].dt.month))
        
        churned = data['churned']
        data.loc[churned, 'months_tenure'] = (
            (data.loc[churned, 'churn_date'].dt.year - data.loc[churned, 'signup_date'].dt.year) * 12 +
            (data.loc[churned, 'churn_date'].dt.month - data.loc[churned, 'signup_date'].dt.month)
        )
        
        cohorts = sorted(data['cohort'].unique())
        retention = []
        
        for cohort in cohorts:
            cohort_data = data[data['cohort'] == cohort]
            cohort_size = len(cohort_data)
            
            row = {'cohort': str(cohort), 'size': cohort_size}
            
            for month in range(months + 1):
                retained = (cohort_data['months_tenure'] >= month).sum()
                row[f'month_{month}'] = retained / cohort_size * 100 if cohort_size > 0 else 0
            
            retention.append(row)
        
        return pd.DataFrame(retention)
    
    retention_table = build_retention_table(filtered_df)
    
    st.subheader("Retention Heatmap")
    
    # Create heatmap data
    retention_matrix = retention_table.iloc[:, 2:].values
    cohort_labels = retention_table['cohort'].values
    month_labels = [f'M{i}' for i in range(retention_matrix.shape[1])]
    
    fig = go.Figure(data=go.Heatmap(
        z=retention_matrix,
        x=month_labels,
        y=cohort_labels,
        colorscale='RdYlGn',
        text=np.round(retention_matrix, 1),
        texttemplate='%{text}%',
        textfont={"size": 10},
        colorbar=dict(title="Retention %")
    ))
    
    fig.update_layout(
        title="Cohort Retention Rate by Month",
        xaxis_title="Months Since Signup",
        yaxis_title="Signup Cohort",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Average Retention Curve")
    avg_retention = retention_table.iloc[:, 2:].mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(avg_retention))),
        y=avg_retention.values,
        mode='lines+markers',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        xaxis_title="Months Since Signup",
        yaxis_title="Average Retention Rate (%)",
        height=400,
        yaxis_range=[0, 105]
    )
    
    st.plotly_chart(fig, use_container_width=True)

elif page == "⚠️ Churn Risk":
    st.markdown('<p class="main-header">Churn Risk Analysis</p>', unsafe_allow_html=True)
    
    active_df = filtered_df[~filtered_df['churned']].copy()
    
    if len(active_df) == 0:
        st.warning("No active users in the selected filters.")
    else:
        # Key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            high_risk = (active_df['risk_level'] == 'High').sum()
            st.metric("High Risk Users", f"{high_risk:,}")
        
        with col2:
            high_risk_mrr = active_df[active_df['risk_level'] == 'High']['mrr'].sum()
            st.metric("High Risk MRR", f"${high_risk_mrr:,.0f}")
        
        with col3:
            at_risk_mrr = active_df[active_df['risk_level'].isin(['High', 'Medium'])]['mrr'].sum()
            st.metric("Total At-Risk MRR", f"${at_risk_mrr:,.0f}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Risk Distribution")
            risk_counts = active_df['risk_level'].value_counts()
            colors = {'High': '#d62728', 'Medium': '#ff7f0e', 'Low': '#2ca02c'}
            color_list = [colors.get(x, '#1f77b4') for x in risk_counts.index]
            
            fig = px.bar(x=risk_counts.index, y=risk_counts.values,
                         labels={'x': 'Risk Level', 'y': 'Number of Users'},
                         color=risk_counts.index,
                         color_discrete_map=colors)
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("MRR by Risk Level")
            mrr_by_risk = active_df.groupby('risk_level')['mrr'].sum()
            color_list = [colors.get(x, '#1f77b4') for x in mrr_by_risk.index]
            
            fig = px.bar(x=mrr_by_risk.index, y=mrr_by_risk.values,
                         labels={'x': 'Risk Level', 'y': 'MRR ($)'},
                         color=mrr_by_risk.index,
                         color_discrete_map=colors)
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.subheader("High-Risk Users")
        high_risk_df = active_df[active_df['risk_level'] == 'High'][
            ['user_id', 'plan', 'monthly_logins', 'feature_adoption', 'team_size', 'mrr', 'days_active']
        ].sort_values('mrr', ascending=False).head(20)
        
        high_risk_df['feature_adoption'] = high_risk_df['feature_adoption'].round(2)
        
        st.dataframe(high_risk_df, use_container_width=True, height=400)
        
        # Download button
        csv = high_risk_df.to_csv(index=False)
        st.download_button(
            label="📥 Download High-Risk Users CSV",
            data=csv,
            file_name="high_risk_users.csv",
            mime="text/csv"
        )

elif page == "💎 Customer Segments":
    st.markdown('<p class="main-header">Customer Lifetime Value Analysis</p>', unsafe_allow_html=True)
    
    # Value segments
    def value_segment(ltv):
        if ltv >= 10000:
            return 'Whale'
        elif ltv >= 2000:
            return 'High Value'
        elif ltv >= 500:
            return 'Medium Value'
        return 'Low Value'
    
    filtered_df['value_segment'] = filtered_df['ltv'].apply(value_segment)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_ltv = filtered_df['ltv'].mean()
        st.metric("Average LTV", f"${avg_ltv:,.0f}")
    
    with col2:
        median_ltv = filtered_df['ltv'].median()
        st.metric("Median LTV", f"${median_ltv:,.0f}")
    
    with col3:
        total_ltv = filtered_df['ltv'].sum()
        st.metric("Total LTV", f"${total_ltv:,.0f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("LTV Distribution")
        fig = px.histogram(filtered_df, x='ltv', nbins=50,
                          labels={'ltv': 'Lifetime Value ($)', 'count': 'Number of Customers'},
                          color_discrete_sequence=['#1f77b4'])
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Average LTV by Plan")
        ltv_by_plan = filtered_df.groupby('plan')['ltv'].mean().sort_values()
        
        fig = px.bar(x=ltv_by_plan.values, y=ltv_by_plan.index, orientation='h',
                     labels={'x': 'Average LTV ($)', 'y': 'Plan'},
                     color_discrete_sequence=['#1f77b4'])
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Segments")
        segment_counts = filtered_df['value_segment'].value_counts()
        
        fig = px.pie(values=segment_counts.values, names=segment_counts.index,
                     color_discrete_sequence=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'])
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Revenue by Segment")
        revenue_by_segment = filtered_df.groupby('value_segment')['ltv'].sum().sort_values(ascending=False)
        
        fig = px.bar(x=revenue_by_segment.index, y=revenue_by_segment.values,
                     labels={'x': 'Segment', 'y': 'Total LTV ($)'},
                     color_discrete_sequence=['#1f77b4'])
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Segment Summary")
    segment_summary = filtered_df.groupby('value_segment').agg({
        'user_id': 'count',
        'ltv': ['mean', 'sum'],
        'churned': 'mean'
    }).round(2)
    segment_summary.columns = ['Customer Count', 'Avg LTV ($)', 'Total LTV ($)', 'Churn Rate']
    st.dataframe(segment_summary, use_container_width=True)

elif page == "📈 Plan Comparison":
    st.markdown('<p class="main-header">Plan Tier Comparison</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Churn Rate by Plan")
        churn_by_plan = filtered_df.groupby('plan')['churned'].mean() * 100
        
        fig = px.bar(x=churn_by_plan.index, y=churn_by_plan.values,
                     labels={'x': 'Plan', 'y': 'Churn Rate (%)'},
                     color_discrete_sequence=['#d62728'])
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Average Engagement by Plan")
        engagement_by_plan = filtered_df.groupby('plan')['monthly_logins'].mean()
        
        fig = px.bar(x=engagement_by_plan.index, y=engagement_by_plan.values,
                     labels={'x': 'Plan', 'y': 'Avg Monthly Logins'},
                     color_discrete_sequence=['#2ca02c'])
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Retention Curves by Plan")
    
    # Calculate retention for each plan
    plans_to_show = ['Free', 'Starter', 'Professional', 'Enterprise']
    colors_map = {'Free': '#1f77b4', 'Starter': '#2ca02c', 'Professional': '#ff7f0e', 'Enterprise': '#d62728'}
    
    fig = go.Figure()
    
    for plan in plans_to_show:
        plan_data = filtered_df[filtered_df['plan'] == plan]
        if len(plan_data) < 10:
            continue
        
        # Simple retention calculation
        months = 12
        retention_rates = []
        for month in range(months + 1):
            retention = (plan_data['days_active'] >= month * 30).mean() * 100
            retention_rates.append(retention)
        
        fig.add_trace(go.Scatter(
            x=list(range(len(retention_rates))),
            y=retention_rates,
            mode='lines+markers',
            name=plan,
            line=dict(width=3, color=colors_map.get(plan)),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        xaxis_title="Months Since Signup",
        yaxis_title="Retention Rate (%)",
        height=500,
        yaxis_range=[0, 105],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Plan Metrics Summary")
    plan_summary = filtered_df.groupby('plan').agg({
        'user_id': 'count',
        'churned': 'mean',
        'monthly_logins': 'mean',
        'feature_adoption': 'mean',
        'ltv': 'mean',
        'mrr': 'sum'
    }).round(2)
    plan_summary.columns = ['Users', 'Churn Rate', 'Avg Logins', 'Avg Feature Adoption', 'Avg LTV', 'Total MRR']
    plan_summary['Churn Rate'] = (plan_summary['Churn Rate'] * 100).round(1).astype(str) + '%'
    
    st.dataframe(plan_summary, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 About")
st.sidebar.info(
    "This dashboard analyzes retention patterns for GrowthHub, "
    "a SaaS marketing automation platform. Data is synthetic but "
    "based on realistic industry patterns."
)

st.sidebar.markdown("### 🔗 Resources")
st.sidebar.markdown("[📊 GitHub Repository](#)")
st.sidebar.markdown("[📓 Full Analysis Notebook](#)")
st.sidebar.markdown("[💼 Portfolio](#)")
