#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Setting up InsightDocs...${NC}"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is not installed. Please install Python 3 first.${NC}"
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}pip3 is not installed. Please install pip3 first.${NC}"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install requirements
echo "Installing required packages..."
pip install streamlit langchain langchain-community faiss-cpu sentence-transformers pdf2image pdfplumber openai pandas

echo -e "${GREEN}Setup completed!${NC}"
echo -e "${GREEN}To start the application, run: ./run.sh${NC}"

# Create run script
cat > run.sh << 'EOL'
#!/bin/bash
source venv/bin/activate

# Start the Streamlit app
streamlit run app.py
EOL

chmod +x run.sh 