#!/bin/bash

# Run RAG Pipeline Streamlit UI
# Usage: ./run_ui.sh

echo "🚀 Starting RAG Pipeline UI..."
echo "Opening at http://localhost:8501"

streamlit run ui/app.py
