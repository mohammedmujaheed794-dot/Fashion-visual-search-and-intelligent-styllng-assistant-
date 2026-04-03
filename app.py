import streamlit as st
import os
import json
from datetime import datetime

# Import existing modules
from src.config import config
from src.stylist import Stylist
from src.feature_extractor import FeatureExtractor
from src.data_loader import DataLoader
from src.search_engine import SearchEngine

# Page Config
st.set_page_config(
    page_title="Fashion Visual Search & Intelligent Styling Assistant",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="collapsed"
)

from src.db import Database

# --- SESSION & DB SETUP ---
if 'user' not in st.session_state:
    st.session_state.user = None
if 'auth_mode' not in st.session_state:
    st.session_state.auth_mode = 'login' # login or signup

# Initialize DB
db = Database()

# --- STYLING CSS ---
st.markdown("""
<style>
    /* Global Clean Dark Theme */
    .stApp {
        background-color: #0E1117;
        font-family: 'Helvetica', sans-serif;
    }
    
    /* Header Navigation */
    .nav-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 2rem;
        background-color: #161B22;
        border-bottom: 1px solid #30363D;
        margin-bottom: 2rem;
        border-radius: 10px;
    }
    .logo-text {
        font-size: 1.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #7928CA, #FF0080);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Action Cards */
    .action-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        transition: transform 0.2s;
        height: 100%;
    }
    .action-card:hover {
        transform: translateY(-5px);
        border-color: #7928CA;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #7928CA, #FF0080);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
        transition: opacity 0.2s;
        width: 100%;
    }
    .stButton>button:hover {
        opacity: 0.9;
        color: white;
    }
    
    /* Secondary Button (Logout/Switch) */
    .secondary-btn button {
        background: transparent !important;
        border: 1px solid #30363D !important;
        color: #c9d1d9 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- FUNCTIONS ---
def login_widget():
    """Sidebar Login Widget"""
    with st.sidebar:
        st.subheader("👤 Account Access")
        if st.session_state.user:
            st.success(f"Logged in as {st.session_state.user}")
            st.info(f"Shopping Preference: **{st.session_state.gender_pref}**")
            if st.button("Log Out"):
                st.session_state.user = None
                st.session_state.gender_pref = "Female" # Reset to default
                st.rerun()
            
            st.markdown("---")
            st.write("📚 **My Saved Looks**")
            user_data = db.get_user(st.session_state.user)
            if not user_data: user_data = {}
            saved = user_data.get('saved_looks', [])
            if saved:
                for look in reversed(saved[-5:]): # Show last 5
                     with st.expander(f"{look['date']} - {look.get('gender', 'N/A')}"):
                        st.write(look['message'])
                        if look['recommendations']:
                            st.image(look['recommendations'][0], width=100)
            else:
                st.info("No saved looks yet.")
                
        else:
            # Create a layout for login form
            placeholder = st.empty()
            with placeholder.form("login"):
                st.markdown("### 🔓 Please Login")
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                
                # Gender Preference for the user (could be saved in DB later)
                gender_pref = st.radio("Shopping Preference", ["Female", "Male"], horizontal=True)
                
                submit = st.form_submit_button("Login")
                
                if submit:
                    if username and password:
                        # In real app, verify hash. 
                        # For demo, just check if user exists in simple JSON DB
                        user_data = db.get_user(username)
                        if user_data:
                            # Check password match
                            if user_data['password'] == password:
                                st.session_state.user = username
                                st.session_state.gender_pref = gender_pref # Save preference
                                st.success(f"Welcome back, {username}!")
                                st.rerun()
                            else:
                                st.error("Incorrect password")
                        else:
                            st.error("User not found.")
                    else:
                        st.error("Please enter credentials")
            
            # Switch to Signup
            if st.button("New here? Create Account"):
                st.session_state.auth_mode = 'signup'
                st.rerun()

def signup_widget():
    with st.sidebar:
        placeholder = st.empty()
        with placeholder.form("signup"):
            st.markdown("### 📝 Create Account")
            new_user = st.text_input("Choose Username")
            new_pass = st.text_input("Choose Password", type="password")
            
            gender_pref = st.radio("Shopping For", ["Female", "Male"], horizontal=True)
            
            submit = st.form_submit_button("Sign Up")
            
            if submit:
                 if new_user and new_pass:
                     if db.get_user(new_user):
                         st.error("User already exists")
                     else:
                         db.add_user(new_user, new_pass, preferences={'gender': gender_pref})
                         st.session_state.user = new_user
                         st.session_state.gender_pref = gender_pref
                         st.success("Account created!")
                         st.rerun()
                 else:
                     st.error("Please fill all fields")
        
        if st.button("Already have an account? Login"):
            st.session_state.auth_mode = 'login'
            st.rerun()

def navbar():
    """Top Navigation Bar"""
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.markdown('<div class="logo-text">Fashion Visual Search & Intelligent Styling Assistant</div>', unsafe_allow_html=True)
    with c2:
        pass # Spacer
    with c3:
        if st.session_state.user:
            st.markdown(f"<div style='text-align:right'>👤 <b>{st.session_state.user}</b></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='text-align:right'>🔴 Not Logged In (Check Sidebar)</div>", unsafe_allow_html=True)

# Init session state for navigation
if 'current_mode' not in st.session_state:
    st.session_state.current_mode = None

def main_interface():
    # 1. Header
    navbar()
    
    # 2. Mode Selection (Home Screen)
    if st.session_state.current_mode is None:
        st.markdown("### 🎯 Choose your goal")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="action-card">
                <h3>� Visual Search</h3>
                <p>Find similar items in your wardrobe</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Start Search", key="btn_search"):
                st.session_state.current_mode = "search"
                st.rerun()

        with col2:
            st.markdown("""
            <div class="action-card">
                <h3>✨ Styling Assistant</h3>
                <p>Get outfit recommendations</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Get Styled", key="btn_style"):
                st.session_state.current_mode = "style"
                st.rerun()
                
        # Show recent saves mini-gallery below?
        return

    # 3. Active Mode UI
    # Back Button
    if st.button("← Back to Home"):
        st.session_state.current_mode = None
        st.rerun()

    mode_title = "Visual Similarity Search" if st.session_state.current_mode == "search" else "Intelligent Styling Assistant"
    st.markdown(f"## {mode_title}")
    
    # Gender Context
    # Default to session state pref if available
    default_ix = 0
    if 'gender_pref' in st.session_state and st.session_state.gender_pref == "Male":
        default_ix = 1
        
    gender = st.selectbox("Gender Preference", ["Female", "Male"], index=default_ix)

    # Input Cards
    c1, c2, c3 = st.columns(3)
    img_file = None
    
    with c1:
        img_file = st.file_uploader("📂 Upload Image", type=['jpg', 'png', 'jpeg'])
    with c2:
        cam_file = st.camera_input("📷 Camera")
        if cam_file: img_file = cam_file
    with c3:
        st.markdown("<br>", unsafe_allow_html=True) # spacer
        if st.button("🎲 Pick Random from Dataset"):
            # Logic to pick random image
            if os.path.exists(config.FILENAMES_PATH):
                import numpy as np
                all_files = np.load(config.FILENAMES_PATH)
                if len(all_files) > 0:
                    random_img = random.choice(all_files)
                    # Create a simple class to mimic the uploaded file interface if needed, 
                    # OR just pass the path. 
                    # For consistency with the rest of the app which expects 'img_file' to be bytes usually,
                    # we will just load it into bytes to be consistent.
                    with open(random_img, "rb") as f:
                        img_file = f.read()
                        
                    # We need to wrap it so it has 'getbuffer' or just handle it below
                    # Actually, let's just use a session state to store the path and handle it 
                    st.session_state.random_image_path = random_img
                    st.rerun()
                else:
                    st.error("Dataset is empty.")
            else:
                st.error("No dataset index found. Please re-index.")
                
    # Handle the random image selection state
    if 'random_image_path' in st.session_state and st.session_state.random_image_path:
        # Load it as if it was uploaded
        try:
             with open(st.session_state.random_image_path, "rb") as f:
                import io
                img_bytes = f.read()
                img_file = io.BytesIO(img_bytes)
                # Force cleanup of state so it doesn't persist forever if they upload something else
                # Actually, we should probably keep it until they upload something else.
                # But for now, let's just set it.
        except Exception as e:
            st.error(f"Error loading random image: {e}")
            del st.session_state.random_image_path

    # Processing
    if img_file:
        # Clear random state if a real upload happens (if we want that behavior)
        # But here 'img_file' is set BY the random logic if it ran.
        
        st.markdown("---")
        
        # Save temp
        temp_path = "temp_query.jpg"
        with open(temp_path, "wb") as f:
            f.write(img_file.getbuffer())
        
        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
            st.image(temp_path, caption="Query Image", use_container_width=True)
            
        with col_res2:
            if not os.path.exists(config.EMBEDDINGS_PATH):
                st.error("System Offline. Please re-index dataset in Admin Settings.")
                return

            # --- MODE SPECIFIC LOGIC ---
            
            # MODE 1: VISUAL SEARCH
            if st.session_state.current_mode == "search":
                with st.spinner("Searching inventory..."):
                    engine = SearchEngine()
                    try:
                        # Apply Gender Filter
                        results = engine.search(temp_path, top_k=10, gender_filter=gender)
                        if results:
                            st.success(f"Found {len(results)} matches ({gender})")
                            cols = st.columns(3)
                            for i, res in enumerate(results):
                                with cols[i % 3]:
                                    st.image(res['image_path'], use_container_width=True)
                                    # Show gender if known for debug
                                    g_label = res.get('gender', 'Unisex')
                                    icon = "⚥"
                                    if g_label.lower() == 'male': icon = "👨"
                                    elif g_label.lower() == 'female': icon = "👩"
                                    
                                    st.caption(f"{icon} {res['category']} | {g_label}")
                        else:
                            st.warning(f"No matches found for {gender}.")
                    except Exception as e:
                        st.error(f"Error: {e}")
        
            # MODE 2: STYLING ASSISTANT
            elif st.session_state.current_mode == "style":
                with st.spinner("Creating outfit..."):
                    stylist = Stylist()
                    try:
                        recs = stylist.get_outfit_recommendations(temp_path, gender=gender)
                        
                        if "error" in recs:
                            st.error(recs['error'])
                        else:
                            st.success(f"Category: **{recs['input_category'].title()}**")
                            st.info(recs['explanation'])
                            
                            st.markdown("#### ✨ Complete the Look")
                            rec_images = []
                            for sugg in recs['styling_suggestions']:
                                st.write(f"**{sugg['category'].title()}**")
                                cols = st.columns(3)
                                for i, item in enumerate(sugg['items']):
                                    with cols[i]:
                                        st.image(item['image_path'], use_container_width=True)
                                        rec_images.append(item['image_path'])
                            
                            # Save Feature (Only for Styling)
                            if st.session_state.user:
                                if st.button("❤️ Save Outfit"):
                                    look = {
                                        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                                        "gender": gender,
                                        "recommendations": rec_images,
                                        "message": recs['explanation']
                                    }
                                    if db.add_saved_look(st.session_state.user, look):
                                        st.toast("Saved!", icon="🎉")
                                    else:
                                        st.error("Failed to save.")
                    except Exception as e:
                        st.error(f"Error: {e}")
                                
# --- APP EXECUTION ---
login_widget()
main_interface()

# Admin Settings (Hidden in Sidebar Bottom for initial setup)
with st.sidebar:
    st.markdown("---")
    with st.expander("🛠️ Admin Settings"):
        st.write("Initializing the AI engine")
        if st.button("Re-Index Dataset"):
            path = config.DATASET_PATH
            st.info(f"Scanning: {path}")
            
            try:
                loader = DataLoader() # Uses config path automatically
                data = loader.validate_dataset() # Returns (path, cat, gender)
                if data:
                    ext = FeatureExtractor()
                    param_paths = [d[0] for d in data]
                    feats, valid = ext.extract_dataset_embeddings(param_paths)
                    
                    # Re-map categories and genders
                    path_to_item = {item[0]: item for item in data}
                    valid_cats = []
                    valid_genders = []
                    
                    for p in valid:
                        item = path_to_item[p]
                        valid_cats.append(item[1])
                        valid_genders.append(item[2])
                    
                    ext.save_embeddings(feats, valid, valid_cats, valid_genders)
                    
                    # Show Stats
                    st.success("Indexing Done!")
                    
                    # Calculate Stats
                    import collections
                    stats = collections.defaultdict(int)
                    for _, _, g in data:
                        stats[g] += 1
                    
                    st.markdown("### 📊 Dataset Statistics")
                    st.json(stats)
                    
                    st.rerun() # Refresh to update random picker availability
                else:
                    st.warning("No images found in data folder.")
            except Exception as e:
                st.error(f"Error: {e}")
