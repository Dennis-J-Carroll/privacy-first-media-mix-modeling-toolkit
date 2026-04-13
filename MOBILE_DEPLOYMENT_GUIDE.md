# 📱 Mobile Web App Deployment Guide

## Quick Start (Local Testing)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all necessary packages including Streamlit.

### 2. Run the App Locally

```bash
streamlit run mmm_mobile_app.py
```

The app will open automatically in your browser at `http://localhost:8501`

### 3. Test on Mobile (Same Wi-Fi Network)

**Step 1:** Find your computer's local IP address

On Mac/Linux:
```bash
ifconfig | grep "inet " | grep -v 127.0.0.1
```

On Windows:
```bash
ipconfig
```

Look for something like `192.168.1.XXX`

**Step 2:** On your phone's browser, navigate to:
```
http://YOUR-LOCAL-IP:8501
```

Example: `http://192.168.1.100:8501`

---

## 🚀 Deploy to Streamlit Cloud (FREE)

### Why Streamlit Cloud?
- ✅ **Free hosting** for public repos
- ✅ **Automatic updates** when you push to GitHub
- ✅ **HTTPS by default** (secure)
- ✅ **Custom URL** (e.g., `yourapp.streamlit.app`)
- ✅ **Works on ALL mobile devices** (no installation needed)

### Step-by-Step Deployment

#### 1. Push to GitHub

```bash
# Make sure you're in the repository root
cd privacy-first-media-mix-modeling-toolkit

# Add all files
git add .

# Commit
git commit -m "Add mobile Streamlit web app"

# Push to GitHub
git push origin main
```

#### 2. Sign Up for Streamlit Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Click **"Sign in with GitHub"**
3. Authorize Streamlit to access your repositories

#### 3. Deploy Your App

1. Click **"New app"**
2. Select your repository: `privacy-first-media-mix-modeling-toolkit`
3. Set the main file path: `mmm_mobile_app.py`
4. Click **"Deploy!"**

#### 4. Wait for Deployment (1-2 minutes)

Streamlit Cloud will:
- ✅ Install all dependencies from `requirements.txt`
- ✅ Start your app
- ✅ Provide a public URL

#### 5. Get Your URL

After deployment, you'll get a URL like:
```
https://privacy-mmm-yourname.streamlit.app
```

**Share this URL** with anyone - it works on all phones!

---

## 📱 Mobile Testing Checklist

Before sharing your app, test these on your phone:

- [ ] **Sliders respond to touch** (try dragging epsilon slider)
- [ ] **Buttons are easy to tap** (44x44 pixel minimum)
- [ ] **Charts load and render** properly
- [ ] **Text is readable** without zooming
- [ ] **No horizontal scrolling** required
- [ ] **Landscape mode works** for charts
- [ ] **Download button works** (CSV export)

### Testing on Different Devices

| Device Type | Recommended Test |
|-------------|------------------|
| iPhone (Safari) | Test slider interaction, chart pinch-zoom |
| Android (Chrome) | Test button taps, CSV downloads |
| iPad (Safari) | Test landscape mode, multi-column layout |
| Android Tablet | Test responsive layout, chart visibility |

---

## 🎨 Customization Options

### Change App Theme

Edit `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#FF6B6B"  # Your brand color
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### Modify Mobile Layout

In `mmm_mobile_app.py`, adjust:

```python
st.set_page_config(
    page_title="Your Title",
    page_icon="🎯",  # Your emoji
    layout="wide",  # or "centered"
    initial_sidebar_state="collapsed"  # Better for mobile
)
```

### Add Your Branding

Replace placeholders in the footer:

```python
st.markdown("""
### 🔗 Learn More
- [Your Website](https://your-website.com)
- [Your Documentation](https://docs.your-site.com)
""")
```

---

## 🔧 Troubleshooting

### Issue: "Module not found" error

**Solution:** Make sure `requirements.txt` includes all dependencies:

```bash
pip freeze | grep -E "(streamlit|numpy|pandas|matplotlib|scipy|seaborn)" > requirements.txt
```

### Issue: App doesn't load on mobile

**Solution:** Check that:
1. Your phone is on the **same Wi-Fi network**
2. Your computer's **firewall allows port 8501**
3. You're using **http://** not **https://** for local testing

### Issue: Charts look too small on mobile

**Solution:** In the app, users can:
- **Pinch to zoom** on charts
- **Rotate to landscape mode** for better viewing
- **Tap the expand icon** (⤢) on charts

### Issue: Streamlit Cloud deployment fails

**Solution:**
1. Check **deployment logs** in Streamlit Cloud dashboard
2. Verify `requirements.txt` has **pinned versions**:
   ```
   streamlit==1.28.0
   numpy==1.24.0
   ```
3. Ensure `advanced_mmm.py` is in the **same directory**

---

## 🎯 Performance Tips

### For Faster Load Times

1. **Cache expensive computations:**

```python
@st.cache_data
def generate_weekly_data():
    # Your data generation code
    pass
```

2. **Limit default data size:**

```python
num_weeks = st.slider("Weeks", 52, 208, 104)  # Default to 104, not 208
```

3. **Optimize image sizes:**

```python
fig, ax = plt.subplots(figsize=(8, 5))  # Smaller = faster
```

---

## 📊 Usage Analytics (Optional)

### Add Google Analytics

1. Create a GA4 property
2. Add to `.streamlit/config.toml`:

```toml
[client]
googleAnalyticsTag = "G-XXXXXXXXXX"
```

### Track User Interactions

In your app:

```python
if st.button("Run Analysis"):
    st.session_state['analyses_run'] = st.session_state.get('analyses_run', 0) + 1
    st.sidebar.metric("Analyses Run", st.session_state['analyses_run'])
```

---

## 🌐 Custom Domain (Advanced)

### Use Your Own Domain

Streamlit Cloud supports custom domains (Pro plan):

1. Upgrade to **Streamlit Cloud Pro**
2. Go to **App Settings → Domains**
3. Add your domain: `mmm.yourdomain.com`
4. Update your DNS **CNAME record**:
   ```
   mmm.yourdomain.com → your-app.streamlit.app
   ```

---

## 🔐 Security Considerations

### Public vs Private Apps

**Public apps** (free):
- ✅ Anyone can access with the URL
- ✅ No login required
- ⚠️ Don't include sensitive data

**Private apps** (Pro):
- 🔒 Password-protected
- 🔒 Google OAuth integration
- 🔒 Restrict to specific email domains

### Best Practices

1. **Never hardcode secrets** in the app
2. **Use Streamlit secrets management**:
   ```python
   import streamlit as st
   api_key = st.secrets["api_key"]
   ```
3. **Validate all user inputs** before processing
4. **Rate limit expensive operations** to prevent abuse

---

## 📚 Additional Resources

### Streamlit Documentation
- [Streamlit Docs](https://docs.streamlit.io/)
- [Mobile Best Practices](https://docs.streamlit.io/library/advanced-features/configuration)
- [Deployment Guide](https://docs.streamlit.io/streamlit-community-cloud/get-started)

### Example Apps
- [Streamlit Gallery](https://streamlit.io/gallery)
- [Mobile-Optimized Examples](https://github.com/streamlit/streamlit/wiki/Gallery)

### Community Support
- [Streamlit Forum](https://discuss.streamlit.io/)
- [GitHub Issues](https://github.com/streamlit/streamlit/issues)

---

## 🎉 Success!

Once deployed, you'll have a **production-ready mobile web app** that:

- ✅ Works on **all phones and tablets**
- ✅ **No installation** required for users
- ✅ **Auto-updates** when you push to GitHub
- ✅ **HTTPS secure** by default
- ✅ **Free hosting** (for public repos)

**Share your URL** and let users explore privacy-first MMM on their phones! 📱🔒

---

## Quick Commands Reference

```bash
# Local development
streamlit run mmm_mobile_app.py

# Find your local IP (Mac/Linux)
ifconfig | grep "inet " | grep -v 127.0.0.1

# Find your local IP (Windows)
ipconfig

# Check app on local network (phone browser)
http://YOUR-LOCAL-IP:8501

# Deploy to GitHub
git add .
git commit -m "Deploy mobile app"
git push origin main
```

---

**Questions?** Open an issue on GitHub or consult the [Streamlit documentation](https://docs.streamlit.io/).

**Happy mobile MMM-ing!** 🚀📊🔒
