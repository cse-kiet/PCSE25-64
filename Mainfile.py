import streamlit as st
from PIL import Image
import io

# Page Config - No sidebar
st.set_page_config(
    page_title="AquaGrowth | Smart Hydroponics",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS with top navigation bar
st.markdown(f"""
<style>
    /* Main app styling */
    .stApp {{
        background: linear-gradient(rgba(245, 247, 250, 0.7), rgba(228, 240, 242, 0.7)), 
                    url("https://media.istockphoto.com/id/591811738/photo/irrigation-system-in-function.jpg?s=612x612&w=0&k=20&c=jWKwu8Di5G958OYj2UDH14FfBM7AJ6MzroXcPiURHS8=");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        font-family: 'Poppins', sans-serif;
    }}
        
    /* Content container for better readability */
    .main-container {{
        background-color: rgba(255, 255, 255, 0.85);
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }}
    
    /* Hide default hamburger menu */
    [data-testid="collapsedControl"] {{
        display: none;
    }}
    
    /* Top navigation bar - updated */
    .top-nav {{
        display: flex;
        justify-content: center;
        background-color: transparent !important;  /* Changed to transparent */
        position: sticky;
        top: 0;
        z-index: 100;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }}
    
    .nav-item {{
        padding: 0.5rem 1.5rem;
        margin: 0 0.5rem;
        color: black !important;
        font-size: 1.2rem !important; /* Adjust size as needed */
        font-weight: 500;
        border-radius: 20px;
        transition: all 0.3s;
        background-color: transparent !important;  /* Transparent background */
        box-shadow: none !important;  /* Remove any shadow/line */
        text-decoration: none !important
    }}
    
    .nav-item:hover {{
        color: #;  /* Change color on hover */
        background-color: rgba(255, 255, 255, 0.7) !important;  /* Slight white tint on hover */
    }}
    
    .nav-item.active {{
        color: # !important;
        font-weight: 600;
        border-bottom: none !important;  /* Remove any bottom border */
    }}
    
    /* Title styling */
    .title {{
        font-size: 3.5rem !important;
        font-weight: 700 !important;
        color: #000000 !important;
        margin-bottom: 0.5rem !important;
    }}
    
    .subtitle {{
        font-size: 2rem !important;
        color: #000000 !important;
        margin-bottom: 1rem !important;
    }}
    
    .header {{
        padding: 2rem 0;
        text-align: center;
    }}
    
    /* Welcome text styling */
    .welcome-text {{
        font-size: 1.8rem;
        color: #2c8a5a;
        margin-bottom: 1.5rem;
        font-weight: 500;
    }}
    
    /* Button styling */
    .cta-button {{
        background: linear-gradient(135deg, #2c8a5a 0%, #4bc0a5 100%) !important;
        color: white !important;
        border: none !important;
        padding: 12px 30px !important;
        font-size: 1.1rem !important;
        border-radius: 30px !important;
        margin-top: 1.5rem !important;
        box-shadow: 0 4px 15px rgba(44, 138, 90, 0.3);
        transition: all 0.3s !important;
    }}
    
    .cta-button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(44, 138, 90, 0.4);
    }}
    
    /* Image styling */
    .responsive-img {{
        max-width: 100%;
        max-height: 390px;
        height: auto;
        width: 1000;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        object-fit: contain; 
    }}
</style>
""", unsafe_allow_html=True)

# Top Navigation Bar
st.markdown("""
<div class="top-nav">
    <a href="/Introduction" class="nav-item active">Main</a>
    <a href="/Dashboard" class="nav-item">Dashboard</a>
    <a href="/Dataset" class="nav-item">Dataset</a>
    <a href="/Research_Work" class="nav-item">Research Work</a>  
    <a href="/About" class="nav-item">About</a>     
</div>
""", unsafe_allow_html=True)

# Main Content
col1, col2 = st.columns([1, 1])
with col1:
    st.markdown('<div class="header">', unsafe_allow_html=True)
    st.markdown('<h1 class="title">Aquagrowth:</h1>', unsafe_allow_html=True)
    st.markdown('<h4 class="subtitle">Where Tech Meets Nature</h4>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 1.1rem; color: #555;">AI-powered solutions for optimal plant growth</p>', unsafe_allow_html=True)
    if st.button("Predict Your Harvest!", key="cta_main", help="Begin your hydroponics journey"):
        st.switch_page("pages/Growth_Predictor.py")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.image(
        "https://media.istockphoto.com/id/615420436/photo/food-production-in-hydroponic-plant-lettuce.jpg?s=612x612&w=0&k=20&c=U2pqQ9YBwG53zbehepR3IwQyHTySk0W1LJcJzbnJkdk=",
        use_container_width=True
    )

# Footer
st.markdown("---")
st.markdown('<p style="text-align: center; color: #666;">© 2025 AquaGrowth | Sustainable Farming Through AI/ML</p>', unsafe_allow_html=True)
