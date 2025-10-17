#!/bin/bash

# Script to run the hybrid embedding pipeline

echo "🚀 Setting up Hybrid PDF Embedding Pipeline"
echo "==========================================="

# Check if we're in the correct directory
if [ ! -d "documents" ]; then
    echo "❌ Documents folder not found. Please run from the root directory."
    exit 1
fi

# Check if we're in the ingest directory, if not, navigate to it
if [ ! -f "ingest/embedder.py" ]; then
    echo "❌ embedder.py not found in ingest directory"
    exit 1
fi

# Install requirements if needed
echo "📦 Installing requirements..."
cd ingest
pip3 install -r requirements.txt

# Run the embedding pipeline
echo "🔄 Running hybrid embedding pipeline..."
python3 embedder.py

echo "✅ Pipeline execution completed!"