# 🎨 Streamlit UI - Complete Setup Guide

## 📋 Summary

Full Streamlit web interface for the RAG Pipeline including:
- Interactive query interface
- Real-time metrics and analytics
- Smart caching with statistics
- Debug console
- Performance monitoring
- Settings & configuration

## 📁 File Structure

```
ui/
├── app.py                    # Main Streamlit app
├── __init__.py              # Package initialization
├── start.py                 # Quick start script
├── components.py            # Reusable UI components
├── requirements.txt         # Streamlit dependencies
├── README.md               # Complete UI documentation
├── SETUP.md                # Quick setup guide
└── pages/
    ├── __init__.py
    ├── analytics.py        # Analytics & charts page
    └── settings.py         # Settings & configuration page

.streamlit/
└── config.toml             # Streamlit configuration
```

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r ui/requirements.txt

# 2. Ensure Ollama is running
ollama serve

# 3. Run the UI
streamlit run ui/app.py

# Open: http://localhost:8501
```

Or use quick start script:
```bash
python ui/start.py
```

## ✨ Features Overview

### 🎯 Main Interface (app.py)
- **Query Input**: Natural language question input
- **Submit Button**: Process query through RAG pipeline
- **Results Display**:
  - Generated answer (green box)
  - Cache status (HIT/MISS badge)
  - Performance metrics (4-column layout)
  - Retrieved documents (tabs for reranked vs all)
  - Query metadata (tokens, model, temperature)

### 📊 Sidebar Configuration
- **Pipeline Settings**: Top-K values, temperature, TTL
- **Model Selection**: Choose Qwen model variants
- **Cache Management**: Enable/disable, clear cache
- **Live Statistics**: Hit rate, latency, token count
- **Query History**: Quick access to recent queries

### 🔧 Advanced Pages
- **📊 Analytics** (`pages/analytics.py`):
  - Performance charts (Plotly)
  - Cache hit/miss pie chart
  - Key metrics (5 cards)
  - Performance recommendations
  - Query history table

- **⚙️ Settings** (`pages/settings.py`):
  - Configuration guide
  - Performance tuning options
  - Troubleshooting tips
  - Developer settings
  - System information

### 🐛 Debug Console
- Raw JSON result inspection
- Full query history table
- Application logs
- System status

## 💾 Session State Management

Using `st.session_state` to persist:
- `pipeline`: RAG Pipeline instance
- `cache_manager`: Cache Manager instance
- `query_history`: List of previous queries
- `current_result`: Latest query result
- `debug_logs`: Application logs
- `error_message`: Error details

## 🎨 Streamlit Configuration

File: `.streamlit/config.toml`
- Theme colors (primary, background, text)
- Client settings (error details, toolbar)
- Logger level
- Server port (8501 default)

## 🔒 Security Features

- ✅ SHA-256 query hashing
- ✅ Local cache only (SQLite)
- ✅ No remote API calls (except Ollama)
- ✅ Session-based state
- ⚠️ For production: add HTTPS, authentication

## 📊 Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Page load | ~2s | Streamlit startup |
| Pipeline init | ~30s | First run only |
| Query (cached) | ~0.01s | SQLite lookup |
| Query (full) | ~11s | Full RAG pipeline |
| UI render | <1s | Result display |

## 🚢 Deployment Options

### Local Development
```bash
streamlit run ui/app.py
```

### Streamlit Cloud
```bash
streamlit cloud deploy
```

### Docker
```dockerfile
FROM python:3.11
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt -r ui/requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "ui/app.py"]
```

### Cloud Platforms
- AWS App Runner
- Google Cloud Run
- Azure Container Instances
- DigitalOcean App Platform
- Heroku

## 🔧 Customization Examples

### Add Custom Metric Card
```python
st.metric("Custom Metric", value, delta=None)
```

### Add Custom Chart
```python
import plotly.express as px
fig = px.bar(data, x="category", y="value")
st.plotly_chart(fig, use_container_width=True)
```

### Add New Page
Create `ui/pages/my_page.py`:
```python
import streamlit as st

st.title("My Custom Page")
# Your content here
```

## 📚 Documentation Files

1. **ui/README.md** - Complete UI documentation
2. **ui/SETUP.md** - Quick setup and first steps
3. **ui/components.py** - Reusable UI components
4. **main README.md** - Full project documentation

## ❓ FAQ

**Q: Can I modify the configuration?**
A: Yes! Use sidebar settings to change retriever_top_k, temperature, etc.

**Q: How do I improve speed?**
A: Enable cache (default), reduce reranker_top_k, use 3B model.

**Q: How do I improve quality?**
A: Increase top_k values, lower temperature, use 7B model.

**Q: Can I export results?**
A: Use Debug → Raw Result JSON section to copy/export.

**Q: Is it production-ready?**
A: Almost! Add authentication, HTTPS, rate limiting for production.

## 🤝 Contributing

Ideas for improvements:
- [ ] Query result export (PDF/JSON)
- [ ] Document upload interface
- [ ] User authentication
- [ ] Real-time streaming responses
- [ ] Query feedback/ratings
- [ ] Advanced search filters

## 📞 Support

- Check **ui/SETUP.md** for quick start
- Check **ui/README.md** for detailed guide
- Use Debug console in UI
- Check Settings page for troubleshooting

---

**Ready to go!** Run `streamlit run ui/app.py` 🚀
