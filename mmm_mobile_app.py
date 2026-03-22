#!/usr/bin/env python3
"""
Privacy-First MMM - Mobile Web App
===================================

Mobile-optimized Streamlit interface for exploring privacy-utility tradeoffs
in Media Mix Modeling with differential privacy.

Features:
- Touch-friendly sliders for privacy parameters
- Real-time model fitting and visualization
- Responsive design for all screen sizes
- Interactive privacy budget exploration

Usage:
    streamlit run mmm_mobile_app.py

Deploy to Streamlit Cloud:
    1. Push to GitHub
    2. Go to https://streamlit.io/cloud
    3. Connect your repo
    4. Deploy!
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Import core MMM functions from new mmm package
from mmm import (
    generate_weekly_data,
    apply_differential_privacy,
    fit_model,
    CONFIG,
    CHANNEL_COLORS,
    geometric_adstock,
    hill_function,
    calculate_marginal_roi
)

# Set page config for mobile optimization
st.set_page_config(
    page_title="Privacy-First MMM",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="collapsed"  # Better for mobile
)

# Custom CSS for mobile optimization
st.markdown("""
<style>
    /* Mobile-friendly styling */
    .stSlider > div > div > div > div {
        font-size: 16px !important;  /* Prevent zoom on mobile */
    }

    /* Larger touch targets */
    .stButton > button {
        min-height: 44px;
        font-size: 18px;
    }

    /* Better spacing on small screens */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Responsive plots */
    .element-container img {
        max-width: 100%;
        height: auto;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.title("🔒 Privacy-First MMM")
st.markdown("**Mobile-Optimized** Media Mix Modeling with Differential Privacy")

# Sidebar for advanced settings (collapsed on mobile)
with st.sidebar:
    st.header("⚙️ Advanced Settings")

    show_technical = st.checkbox("Show technical details", value=False)
    num_weeks = st.slider("Weeks of data", 52, 208, 104, help="Number of weeks to simulate")
    noise_std = st.slider("Baseline noise (σ)", 500, 2000, 1000, help="Natural data variability")

# Main content
st.header("🎛️ Privacy Controls")

# Privacy parameter slider (main control)
epsilon = st.slider(
    "Privacy Budget (ε)",
    min_value=0.1,
    max_value=10.0,
    value=1.0,
    step=0.1,
    help="Lower ε = More Privacy but Less Accuracy"
)

# Visual feedback on privacy level
privacy_level = "🔐 High Privacy" if epsilon < 1.0 else "⚖️ Balanced" if epsilon < 5.0 else "📊 Low Privacy"
st.metric("Privacy Level", privacy_level, f"ε = {epsilon}")

# Explanation (collapsible on mobile)
with st.expander("ℹ️ What does ε mean?"):
    st.markdown(f"""
    **Current setting: ε = {epsilon}**

    - **ε < 1.0**: Strong privacy protection, but more noise in results
    - **ε ≈ 1.0**: Good balance between privacy and utility (recommended)
    - **ε > 5.0**: Weak privacy, minimal noise

    **Differential Privacy** ensures individual-level data cannot be extracted,
    even if an attacker has access to the aggregated results.
    """)

# Run analysis button
if st.button("🚀 Run Analysis", use_container_width=True):

    with st.spinner("Generating synthetic data..."):
        # Update config
        CONFIG["num_weeks"] = num_weeks
        CONFIG["epsilon"] = epsilon
        CONFIG["noise_std"] = noise_std

        # Generate data
        df_original = generate_weekly_data()

    st.success("✅ Data generated!")

    with st.spinner("Applying differential privacy..."):
        # Apply privacy
        df_private = apply_differential_privacy(df_original, epsilon)

    st.success(f"✅ Privacy applied (ε = {epsilon})")

    with st.spinner("Fitting MMM model..."):
        # Fit model
        results = fit_model(df_private)

    st.success("✅ Model fitted!")

    # Store results in session state for later use
    st.session_state['results'] = results
    st.session_state['df_original'] = df_original
    st.session_state['df_private'] = df_private
    st.session_state['epsilon'] = epsilon

# Display results if available
if 'results' in st.session_state:

    results = st.session_state['results']
    df_original = st.session_state['df_original']
    df_private = st.session_state['df_private']
    epsilon_used = st.session_state['epsilon']

    st.header("📊 Results")

    # Key metrics in columns (mobile-friendly)
    st.subheader("💰 Return on Investment (ROI)")

    channels = CONFIG["channels"]
    cols = st.columns(len(channels))

    for i, channel in enumerate(channels):
        with cols[i]:
            fitted = results['fitted_params'][channel]
            true = CONFIG['true_params'][channel]

            # Calculate mROI at current spend level
            avg_spend = df_private[f"spend_{channel}"].mean()
            mroi = calculate_marginal_roi(
                avg_spend,
                fitted['adstock_decay'],
                fitted['hill_alpha'],
                fitted['hill_K'],
                fitted['hill_beta']
            )

            st.metric(
                channel,
                f"${mroi:.2f}",
                delta=None,
                help=f"Marginal ROI: Revenue per additional $1 spent"
            )

    # Model accuracy
    st.subheader("🎯 Model Accuracy")
    r_squared = results['r_squared']

    accuracy_color = "🟢" if r_squared > 0.8 else "🟡" if r_squared > 0.6 else "🔴"
    st.metric(
        "Model Fit (R²)",
        f"{accuracy_color} {r_squared:.3f}",
        help="How well the model explains revenue variation (0-1 scale)"
    )

    # Privacy impact explanation
    if show_technical:
        st.info(f"""
        **Privacy Impact Analysis** (ε = {epsilon_used})

        - Added noise scale: {1.0/epsilon_used:.2f}x sensitivity
        - Privacy guarantee: Individual contributions protected with ε = {epsilon_used}
        - Model accuracy (R²): {r_squared:.3f}
        """)

    # Parameter comparison table
    st.subheader("📋 Estimated vs True Parameters")

    param_data = []
    for channel in channels:
        fitted = results['fitted_params'][channel]
        true = CONFIG['true_params'][channel]

        for param_name in ['adstock_decay', 'hill_alpha', 'hill_K', 'hill_beta']:
            param_data.append({
                'Channel': channel,
                'Parameter': param_name,
                'True': true[param_name],
                'Estimated': fitted[param_name],
                'Error %': abs(fitted[param_name] - true[param_name]) / true[param_name] * 100
            })

    param_df = pd.DataFrame(param_data)

    # Format for mobile display
    st.dataframe(
        param_df.style.format({
            'True': '{:.2f}',
            'Estimated': '{:.2f}',
            'Error %': '{:.1f}%'
        }).background_gradient(subset=['Error %'], cmap='RdYlGn_r'),
        use_container_width=True,
        height=300
    )

    # Visualizations
    st.header("📈 Visualizations")

    # Response curves
    st.subheader("Channel Response Curves")

    fig, axes = plt.subplots(1, len(channels), figsize=(12, 4))
    if len(channels) == 1:
        axes = [axes]

    for i, channel in enumerate(channels):
        ax = axes[i]

        fitted = results['fitted_params'][channel]
        true = CONFIG['true_params'][channel]

        # Generate spend range
        max_spend = df_private[f"spend_{channel}"].max() * 1.5
        spend_range = np.linspace(0, max_spend, 100)

        # Apply adstock
        adstock_fitted = geometric_adstock(spend_range, fitted['adstock_decay'])
        adstock_true = geometric_adstock(spend_range, true['adstock_decay'])

        # Apply saturation
        revenue_fitted = hill_function(
            adstock_fitted,
            fitted['hill_alpha'],
            fitted['hill_K'],
            fitted['hill_beta']
        )
        revenue_true = hill_function(
            adstock_true,
            true['hill_alpha'],
            true['hill_K'],
            true['hill_beta']
        )

        # Plot
        ax.plot(spend_range, revenue_true, 'k--', label='True', linewidth=2)
        ax.plot(spend_range, revenue_fitted, 'b-', label='Estimated', linewidth=2)
        ax.set_xlabel('Spend ($)', fontsize=10)
        ax.set_ylabel('Revenue Contribution ($)', fontsize=10)
        ax.set_title(channel, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Predicted vs Actual
    st.subheader("Model Predictions")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Calculate predictions
    predicted = results['fitted_params']['predictions']
    actual = df_private['revenue'].values

    # Scatter plot
    ax.scatter(actual, predicted, alpha=0.5, s=50)

    # Perfect prediction line
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--',
            label='Perfect Prediction', linewidth=2)

    ax.set_xlabel('Actual Revenue ($)', fontsize=12)
    ax.set_ylabel('Predicted Revenue ($)', fontsize=12)
    ax.set_title(f'Predicted vs Actual (R² = {r_squared:.3f})',
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)
    plt.close()

    # Privacy-Utility comparison
    if show_technical:
        st.subheader("Privacy Impact on Data")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Original vs Private revenue
        ax = axes[0]
        weeks = range(1, len(df_original) + 1)
        ax.plot(weeks, df_original['revenue'], 'k-', label='Original', alpha=0.7, linewidth=2)
        ax.plot(weeks, df_private['revenue'], 'b-', label=f'Private (ε={epsilon_used})', alpha=0.7, linewidth=2)
        ax.set_xlabel('Week', fontsize=10)
        ax.set_ylabel('Revenue ($)', fontsize=10)
        ax.set_title('Privacy Impact on Revenue', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # Noise distribution
        ax = axes[1]
        noise = df_private['revenue'] - df_original['revenue']
        ax.hist(noise, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No Noise')
        ax.set_xlabel('Added Noise ($)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'Noise Distribution (ε={epsilon_used})', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Download results
    st.header("💾 Export Results")

    # Create CSV
    csv_buffer = BytesIO()
    param_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    st.download_button(
        label="📥 Download Parameter Summary (CSV)",
        data=csv_buffer,
        file_name=f"mmm_results_epsilon_{epsilon_used}.csv",
        mime="text/csv",
        use_container_width=True
    )

# Footer with instructions
st.markdown("---")
st.markdown("""
### 📱 Mobile Tips
- **Pinch to zoom** on charts for details
- **Swipe left/right** to see all channels
- **Tap ℹ️** for explanations
- **Use landscape mode** for better chart viewing

### 🚀 Deploy Your Own
1. Fork this repo on GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your repo
4. Share the URL!

### 🔗 Learn More
- [Differential Privacy Guide](https://github.com/your-repo/privacy_parameter_guide.md)
- [Full Documentation](https://github.com/your-repo/README.md)
""")

st.caption("Built with ❤️ using Streamlit | Privacy-First by Design")
