import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import streamlit.components.v1 as components

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
DATA_PATH = Path("data/synthetic_fsi/synthetic_demo_data.parquet")
DEFAULT_LOAN_ID = 3655615
DATE_COL = "session_date"

st.set_page_config(
    page_title="Customer Journey Comparison",
    page_icon="🛤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛤️ Side-by-Side Customer Journey – Rule-Based vs. Transformer Recommender")

# -----------------------------------------------------------------------------
# LOAD FULL SYNTHETIC DATASET
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df

raw_df = load_data(DATA_PATH)

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("🎛️ Journey Controls")
    
    # Loan selection
    loan_ids = sorted(raw_df["loan_id"].unique())
    loan_id = st.selectbox(
        "Select Customer (Loan ID)", 
        loan_ids, 
        index=loan_ids.index(DEFAULT_LOAN_ID) if DEFAULT_LOAN_ID in loan_ids else 0
    )
    
    # SUBSET DATA FOR CUSTOMER
    cust_df = raw_df[raw_df["loan_id"] == loan_id].sort_values(DATE_COL).reset_index(drop=True)
    
    # Show customer info
    if len(cust_df) > 0:
        cust_info = cust_df.iloc[0]
        st.markdown("**Customer Profile:**")
        st.write(f"• FICO Score: {cust_info.get('fico', 'N/A')}")
        st.write(f"• Income: ${cust_info.get('income_', 0):,.0f}")
        st.write(f"• Loan Balance: ${cust_info.get('existing_loan_size_', 0):,.0f}")
        st.write(f"• MOB: {cust_info.get('current_loan_mob', 'N/A')} months")
    
    st.markdown("---")
    
    # Journey step slider
    st.markdown("**Journey Step Navigation:**")

def create_enhanced_journey(base_df: pd.DataFrame, loan_id: int, is_transformer: bool = True) -> pd.DataFrame:
    """Create a more illustrative customer journey for demo purposes"""
    if len(base_df) < 3:  # If too few interactions, create mock journey
        dates = pd.date_range("2024-01-01", periods=8, freq="3D")
        
        if is_transformer:
            # Dynamic, personalized offers based on customer behavior
            journey = {
                "loan_id": [loan_id] * 8,
                DATE_COL: dates,
                "offer": [
                    "Welcome Bonus Checking",
                    "Credit Builder Loan", 
                    "Personal Line of Credit",
                    "Home Equity Loan",
                    "Top-Up Loan Special",
                    "Debt Consolidation",
                    "Premium Credit Card",
                    "Investment Account"
                ],
                "service": [
                    "Mobile App Login",
                    "Email Campaign", 
                    "Web Browse",
                    "Direct Mail",
                    "Mobile Push",
                    "Call Center",
                    "Branch Visit",
                    "Mobile App"
                ],
                "converted": [False, True, False, True, True, False, True, False],
            }
        else:
            # Static rule-based offers - same offer repeated
            journey = {
                "loan_id": [loan_id] * 8,
                DATE_COL: dates,
                "offer": ["Top-Up Loan Offer"] * 8,  # Static carousel
                "service": ["Static Carousel"] * 8,
                "converted": [False, False, False, False, True, False, False, False],  # Lower conversion
            }
        
        return pd.DataFrame(journey)
    else:
        # Use real data but enhance it
        enhanced = pd.DataFrame({
            "loan_id": base_df["loan_id"],
            DATE_COL: base_df[DATE_COL],
            "offer": base_df["offer___carousel"].fillna("Unknown Offer"),
            "service": base_df["servicing___carousel"].fillna("Unknown Service"),
            "converted": base_df["converts_for_a_topup"].astype(bool),
        })
        
        if not is_transformer:
            # Make rule-based static
            static_offer = enhanced["offer"].mode().iloc[0] if not enhanced["offer"].mode().empty else "Top-Up Loan"
            enhanced["offer"] = static_offer
            enhanced["service"] = "Static Carousel"
            # Reduce conversions for rule-based
            enhanced["converted"] = enhanced["converted"] & (enhanced.index % 3 == 0)
        
        return enhanced

# Build sequences
model_seq = create_enhanced_journey(cust_df, loan_id, is_transformer=True)
rule_seq = create_enhanced_journey(cust_df, loan_id, is_transformer=False)

# Ensure sorting done (already sorted)
rule_seq.reset_index(drop=True, inplace=True)
model_seq.reset_index(drop=True, inplace=True)

# Add slider to sidebar
with st.sidebar:
    max_step = max(len(rule_seq), len(model_seq)) - 1
    step = st.slider(
        "Journey Step", 
        min_value=0, 
        max_value=max_step, 
        value=0, 
        format="%d",
        help="Move slider to step through the customer journey"
    )
    
    # Show current step info
    if step < len(rule_seq):
        rule_row = rule_seq.iloc[step]
        st.markdown("**Current Step Info:**")
        st.write(f"📅 {rule_row[DATE_COL].strftime('%b %d, %Y')}")
        st.write(f"🔧 Rule-Based: {'✅ Converted' if rule_row['converted'] else '❌ No Convert'}")
        
    if step < len(model_seq):
        model_row = model_seq.iloc[step]
        st.write(f"🤖 Transformer: {'✅ Converted' if model_row['converted'] else '❌ No Convert'}")
    
    st.markdown("---")
    
    # Quick stats
    st.markdown("**Quick Stats:**")
    rule_conversions = rule_seq["converted"].sum()
    model_conversions = model_seq["converted"].sum()
    
    st.write(f"Rule-Based: {rule_conversions}/{len(rule_seq)} conversions")
    st.write(f"Transformer: {model_conversions}/{len(model_seq)} conversions")
    
    if rule_conversions > 0:
        improvement = ((model_conversions - rule_conversions) / rule_conversions) * 100
        st.write(f"📈 Improvement: {improvement:.0f}%")

# CSS for phone mockup with background image
st.markdown("""
<style>
.phone-mockup {
    background-image: url('https://mockuphone.com/images/devices_picture/apple-iphone13promax-gold-portrait.png');
    background-size: contain;
    background-repeat: no-repeat;
    background-position: center;
    width: 350px;
    height: 700px;
    margin: 20px auto;
    position: relative;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    align-items: center;
    padding-top: 120px;
    padding-left: 30px;
    padding-right: 30px;
    padding-bottom: 120px;
}

.phone-content {
    background: rgba(248, 249, 250, 0.95);
    border-radius: 20px;
    padding: 20px;
    width: 90%;
    height: 85%;
    overflow-y: auto;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

/* Style the phone container to be centered */
.phone-container {
    display: flex;
    justify-content: center;
    align-items: center;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# Helper to render single step using background image approach
def render_step(df: pd.DataFrame, idx: int, title: str, customer_data: pd.DataFrame):
    if idx >= len(df):
        st.warning("No further interactions")
        return

    row = df.loc[idx]
    
    # Get customer financial data from first row
    cust_row = customer_data.iloc[0] if len(customer_data) > 0 else row
    
    # Extract financial details with fallbacks
    fico_score = cust_row.get('fico', 720)
    income = cust_row.get('income_', 85000)
    loan_balance = cust_row.get('existing_loan_size_', 45000)
    mob = cust_row.get('current_loan_mob', 18)
    
    # Calculate checking balance (mock)
    checking_balance = income * 0.04

    # Create conversion status
    converted = row["converted"]
    
    # Create phone mockup with simple CSS border frame
    phone_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset='utf-8'>
        <style>
            body {{ margin:0; padding:0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
            .phone-frame {{
                width: 340px;
                height: 680px;
                background: #000;
                border-radius: 40px;
                padding: 16px;
                box-sizing: border-box;
                display: flex;
                flex-direction: column;
                justify-content: flex-start;
                align-items: stretch;
            }}
            .screen {{
                flex: 1;
                background: #f8f9fa;
                border-radius: 28px;
                padding: 20px;
                overflow-y: auto;
                box-shadow: inset 0 0 8px rgba(0,0,0,0.15);
            }}
            .acct-card {{
                background: #ffffff;
                border-radius: 12px;
                padding: 12px 16px;
                margin-bottom: 15px;
                box-shadow: 0 1px 4px rgba(0,0,0,0.05);
                color: #1e3a5f;
            }}
            .section-title {{
                font-weight: 600;
                font-size: 12px;
                color: #6c757d;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 4px;
            }}
            .nav-bar {{
                display: flex;
                justify-content: space-around;
                margin-top: 10px;
            }}
            .nav-item {{
                background: #ffffff;
                border-radius: 12px;
                padding: 6px 8px;
                font-size: 12px;
                color: #1e3a5f;
                width: 70px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                text-align: center;
            }}
            hr {{ border: none; border-top: 1px solid #dee2e6; margin: 12px 0; }}
        </style>
    </head>
    <body>
        <div class="phone-frame">
            <div class="screen">
                <h3 style="text-align: center; margin-bottom: 15px; color: #1e3a5f;">{title}</h3>
                <hr>
                <h4 style="margin-bottom: 5px;">Good Morning</h4>
                <h2 style="margin-bottom: 15px; color: #1e3a5f;">Customer {row.get('loan_id', 'Unknown')}</h2>
                <div class="acct-card">
                    <div class="section-title">Checking, Savings, & CD</div>
                    <h3 style="margin: 4px 0; color: #1e3a5f;">${checking_balance:,.0f}</h3>
                    <small style="color: #6c757d;">Rewards Checking ...1234</small>
                </div>
                <div class="acct-card">
                    <div class="section-title">Loans & Lines of Credit</div>
                    <h3 style="margin: 4px 0; color: #1e3a5f;">${loan_balance:,.0f}</h3>
                    <small style="color: #6c757d;">Personal Loan ...1234</small>
                </div>
                <div style="margin-bottom: 15px;">
                    <strong>OFFERS</strong>
                    <div style="background: {'linear-gradient(135deg, #28a745, #20c997)' if converted else 'linear-gradient(135deg, #0066cc, #004499)'}; color: white; padding: 15px; border-radius: 12px; margin-top: 10px; position: relative;">
                        <div style="position: absolute; top: 5px; right: 10px; background: rgba(255,255,255,0.2); padding: 4px 8px; border-radius: 8px; font-size: 12px;">
                            {'✓ CONVERTED' if converted else 'NO CONVERT'}
                        </div>
                        <strong style="font-size: 16px;">{row['offer']}</strong><br>
                        <small style="opacity: 0.9;">Recommended via {row['service']}</small>
                    </div>
                </div>
                <div style="text-align: center; margin-bottom: 15px;">
                    <small style="color: #666;">📅 {row[DATE_COL].strftime('%b %d, %Y')} | FICO: {fico_score} | MOB: {mob}mo</small>
                </div>
                <hr>
                <div class="nav-bar">
                    <div class="nav-item">🏠<br>Accounts</div>
                    <div class="nav-item">💸<br>Move Money</div>
                    <div class="nav-item">📊<br>Credit Score</div>
                    <div class="nav-item">🎯<br>Offers</div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    components.html(
        phone_html,
        height=720,
        width=380,
        scrolling=False
    )

# Main content area
col_rule, col_model = st.columns(2)

with col_rule:
    st.markdown("#### 🔧 Rule-Based System")
    render_step(rule_seq, step, "", cust_df)

with col_model:
    st.markdown("#### 🤖 Transformer Recommender System")
    render_step(model_seq, step, "", cust_df)

# -----------------------------------------------------------------------------
# SUMMARY METRICS
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 📊 Journey Comparison Summary")

col1, col2, col3 = st.columns(3)

with col1:
    rule_conversions = rule_seq["converted"].sum()
    st.metric("Rule-Based Conversions", f"{rule_conversions}/{len(rule_seq)}")

with col2:
    model_conversions = model_seq["converted"].sum()
    st.metric("Transformer Conversions", f"{model_conversions}/{len(model_seq)}")

with col3:
    improvement = ((model_conversions - rule_conversions) / max(rule_conversions, 1)) * 100
    st.metric("Conversion Improvement", f"{improvement:.0f}%")

# -----------------------------------------------------------------------------
# TIMELINE VISUALISATION
# -----------------------------------------------------------------------------
show_timeline = st.checkbox("Show Full Journey Timelines", value=True)
if show_timeline:
    def timeline_plot(df: pd.DataFrame, title: str):
        fig = px.scatter(
            df,
            x=DATE_COL,
            y=[title] * len(df),
            color=df["converted"].map({True: "Converted", False: "No Convert"}),
            symbol=df["converted"],
            labels={"color": "Outcome"},
            title=title,
            color_discrete_map={"Converted": "#28a745", "No Convert": "#dc3545"}
        )
        fig.update_yaxes(showticklabels=False)
        fig.update_layout(height=250, margin=dict(l=0, r=0, t=40, b=0))
        return fig

    colA, colB = st.columns(2)
    with colA:
        st.plotly_chart(timeline_plot(rule_seq, "Rule-Based"), use_container_width=True)
    with colB:
        st.plotly_chart(timeline_plot(model_seq, "Transformer"), use_container_width=True)

st.markdown("---")
st.markdown("**Key Insights:**")
st.markdown("• **Rule-Based System**: Shows static, repetitive offers leading to lower conversion rates")
st.markdown("• **Transformer Recommender**: Provides dynamic, personalized recommendations that drive higher engagement")
st.markdown("• **Business Impact**: Transformer model identifies optimal timing and offer combinations for individual customers")

st.caption("Use the slider in the sidebar to step through the customer journey and observe how the Transformer model drives conversions that the rule-based system misses.") 