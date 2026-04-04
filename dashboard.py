"""
Monitoring Dashboard for AI Security Gateway
Real-time security metrics and audit log viewer
"""

import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import time
from datetime import datetime, timedelta
import json

# Configuration
API_URL = "http://localhost:8080"

st.set_page_config(
    page_title="AI Security Gateway Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .blocked-card {
        background-color: #ffebee;
        border-left: 4px solid #dc143c;
    }
    .safe-card {
        background-color: #e8f5e9;
        border-left: 4px solid #2e8b57;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
    }
    .stAlert {
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def fetch_metrics():
    """Fetch metrics from gateway"""
    try:
        response = requests.get(f"{API_URL}/metrics", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def fetch_health():
    """Fetch health status"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def run_evaluation():
    """Run security evaluation"""
    try:
        response = requests.post(f"{API_URL}/evaluate", timeout=30)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def main():
    st.markdown('<div class="main-header">🛡️ AI Security Gateway Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select View",
        ["📊 Overview", "⚡ Performance", "🧪 Evaluation", "🔍 Test Prompt"]
    )
    
    # Auto-refresh checkbox
    auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=True)
    
    if auto_refresh and page == "📊 Overview":
        time.sleep(5)
        st.rerun()
    
    if page == "📊 Overview":
        show_overview()
    elif page == "⚡ Performance":
        show_performance()
    elif page == "🧪 Evaluation":
        show_evaluation()
    elif page == "🔍 Test Prompt":
        show_test_prompt()

def show_overview():
    """Main dashboard overview"""
    
    # Fetch data
    metrics = fetch_metrics()
    health = fetch_health()
    
    if not metrics or not health:
        st.error("❌ Gateway not reachable. Is it running on port 8080?")
        return
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value">{metrics["performance"]["total_requests"]:,}</div>
            <div>Total Requests</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        blocked_rate = metrics["performance"]["blocked_rate"] * 100
        color = "#dc143c" if blocked_rate > 10 else "#2e8b57"
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value" style="color: {color}">{blocked_rate:.1f}%</div>
            <div>Block Rate</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        p99 = metrics["performance"]["p99"]
        color = "#dc143c" if p99 > 100 else "#2e8b57"
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value" style="color: {color}">{p99:.1f}ms</div>
            <div>P99 Latency</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        model_status = "✅ Loaded" if health["model_loaded"] else "⚠️ Fallback"
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value">{model_status}</div>
            <div>Model Status</div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.divider()
    
    # Category breakdown
    st.subheader("24h Category Distribution")
    
    categories = metrics.get("categories_24h", {})
    if categories:
        df = pd.DataFrame([
            {"Category": cat, "Count": data["count"], "Avg Score": data["avg_score"]}
            for cat, data in categories.items()
        ])
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(df, values="Count", names="Category", 
                        title="Blocked by Category")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(df, x="Category", y="Avg Score",
                        title="Average Risk Score by Category")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet. Send some requests through the gateway.")
    
    # System metrics
    st.divider()
    st.subheader("System Health")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cpu = metrics["system"]["cpu_percent"]
        st.metric("CPU Usage", f"{cpu}%", delta=None)
        st.progress(min(cpu / 100, 1.0))
    
    with col2:
        mem = metrics["system"]["memory_percent"]
        st.metric("Memory Usage", f"{mem}%", delta=None)
        st.progress(min(mem / 100, 1.0))

def show_performance():
    """Detailed performance metrics"""
    st.header("⚡ Performance Metrics")
    
    metrics = fetch_metrics()
    if not metrics:
        st.error("Gateway not reachable")
        return
    
    perf = metrics["performance"]
    
    # Latency percentiles
    st.subheader("Latency Percentiles")
    
    latency_data = {
        "p50": perf["p50"],
        "p95": perf["p95"],
        "p99": perf["p99"],
        "mean": perf["mean"],
        "min": perf["min"],
        "max": perf["max"]
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gauge chart for P99
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = perf["p99"],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "P99 Latency (ms)"},
            delta = {'reference': 50, 'decreasing': {'color': "green"}},
            gauge = {'axis': {'range': [None, 200]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgreen"},
                        {'range': [50, 100], 'color': "yellow"},
                        {'range': [100, 200], 'color': "salmon"}],
                    'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75, 'value': 100}}))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Latency distribution table
        st.write("Latency Distribution")
        df = pd.DataFrame([latency_data]).T
        df.columns = ["Value (ms)"]
        st.dataframe(df, use_container_width=True)
    
    # Target: <50ms p99
    if perf["p99"] <= 50:
        st.success("✅ Meeting target: P99 < 50ms")
    else:
        st.warning(f"⚠️ Missing target. P99 is {perf['p99']:.1f}ms (target: <50ms)")

def show_evaluation():
    """Run and display evaluation results"""
    st.header("🧪 Security Evaluation")
    
    st.write("Run evaluation against HarmBench-style test cases")
    
    if st.button("🚀 Run Evaluation", type="primary"):
        with st.spinner("Running tests..."):
            results = run_evaluation()
        
        if results:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                accuracy = results["accuracy"] * 100
                st.metric("Accuracy", f"{accuracy:.1f}%")
            
            with col2:
                st.metric("Passed", results["passed"])
            
            with col3:
                st.metric("Failed", results["failed"])
            
            # Detailed results
            st.subheader("Test Results")
            
            df = pd.DataFrame(results["results"])
            
            # Color code rows
            def highlight_correct(val):
                if isinstance(val, bool):
                    return 'background-color: #e8f5e9' if val else 'background-color: #ffebee'
                return ''
            
            st.dataframe(
                df.style.applymap(highlight_correct, subset=['correct']),
                use_container_width=True
            )
            
            # Pass/Fail chart
            fig = px.pie(
                names=["Passed", "Failed"],
                values=[results["passed"], results["failed"]],
                title="Test Results",
                color=["Passed", "Failed"],
                color_discrete_map={"Passed": "#2e8b57", "Failed": "#dc143c"}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            if accuracy >= 80:
                st.success("✅ Evaluation PASSED (>=80% accuracy)")
            else:
                st.error("❌ Evaluation FAILED (<80% accuracy)")
        else:
            st.error("Failed to run evaluation")

def show_test_prompt():
    """Test individual prompts"""
    st.header("🔍 Test Prompt Security")
    
    prompt = st.text_area(
        "Enter prompt to test:",
        height=150,
        placeholder="Enter a prompt to check security..."
    )
    
    backend_model = st.text_input("Backend Model", value="llama3.2:latest")
    
    if st.button("🔍 Analyze", type="primary") and prompt:
        with st.spinner("Analyzing..."):
            # Send to gateway
            try:
                response = requests.post(
                    f"{API_URL}/v1/chat/completions",
                    json={
                        "model": backend_model,
                        "messages": [{"role": "user", "content": prompt}]
                    },
                    timeout=30
                )
                
                result = response.json()
                security = result.get("security", {})
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    blocked = security.get("blocked", False)
                    st.metric("Blocked", "YES" if blocked else "NO")
                    if blocked:
                        st.error("🚫 Request blocked")
                    else:
                        st.success("✅ Request allowed")
                
                with col2:
                    score = security.get("score", 0)
                    st.metric("Risk Score", f"{score:.2%}")
                
                with col3:
                    category = security.get("category", "unknown")
                    st.metric("Category", category)
                
                # Full response
                st.subheader("Response")
                if blocked:
                    st.info(result["choices"][0]["message"]["content"])
                else:
                    st.write(result["choices"][0]["message"]["content"][:500])
                
                # Raw JSON
                with st.expander("Raw Response"):
                    st.json(result)
                    
            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Make sure the gateway (port 8080) and backend (Ollama) are running")

if __name__ == "__main__":
    main()
