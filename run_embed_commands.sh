#!/bin/bash
###############################################################################
# ZebOS Commands Embedder Runner
# 
# This script embeds ZebOS commands and chapters into the vector database
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         ZebOS Commands & Chapters Embedder                     ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}❌ Virtual environment not found!${NC}"
    echo -e "${YELLOW}Please create it first: python3 -m venv .venv${NC}"
    exit 1
fi

# Activate virtual environment
echo -e "${GREEN}🔄 Activating virtual environment...${NC}"
source .venv/bin/activate

# Check if required files exist
if [ ! -f "zebos_commands.json" ]; then
    echo -e "${RED}❌ zebos_commands.json not found!${NC}"
    exit 1
fi

if [ ! -f "zebos_chapters.json" ]; then
    echo -e "${RED}❌ zebos_chapters.json not found!${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Found ZebOS JSON files${NC}"
echo ""

# Create output directory
mkdir -p database/commands

# Run the embedder
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}🚀 Starting embedding process...${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""

python3 ingest/zebos_commands_embedder.py \
    --commands zebos_commands.json \
    --chapters zebos_chapters.json \
    --output database/commands

EMBED_EXIT_CODE=$?

echo ""
if [ $EMBED_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║              ✅ EMBEDDING COMPLETED SUCCESSFULLY                ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${GREEN}📊 Results saved to: database/commands/${NC}"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo -e "  1. Test the embeddings with a search query"
    echo -e "  2. Update your RAG pipeline to use MultiIndexRetriever"
    echo -e "  3. Run the test script: python3 agent/multi_index_retriever.py"
    echo ""
else
    echo -e "${RED}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                ❌ EMBEDDING FAILED                              ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${RED}Please check the error messages above${NC}"
    exit 1
fi

# Deactivate virtual environment
deactivate
