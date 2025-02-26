# InsightDocs

InsightDocs is an intelligent document management and QA system that allows organizations to upload their documents and interact with them through a natural language interface.

## Features

- **Document Management**
  - Upload and process PDF documents
  - Organize documents by categories
  - Add metadata and descriptions
  - Track document statistics

- **Intelligent QA**
  - Natural language queries
  - Context-aware responses
  - Source citations
  - Chat interface

- **Admin Portal**
  - Secure admin access
  - Document management
  - Usage analytics
  - Model configuration

- **Usage Tracking**
  - Token usage monitoring
  - Cost tracking
  - Usage analytics
  - Export capabilities

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/ashishkhar/InsightDocs.git
cd InsightDocs
```

2. Run setup script:
```bash
./setup.sh
```

3. Start the application:
```bash
./run.sh
```

4. Open in browser:
```
http://localhost:8501
```

## First Time Setup

1. Create admin account when first launching the app
2. Save the access token for admin portal access
3. Upload your first documents
4. Start asking questions!

## Usage

### User Portal
- Access the main interface at `http://localhost:8501`
- Upload documents and ask questions
- View source documents for answers
- Track conversation history

### Admin Portal
- Access at `http://localhost:8501/?access_token=YOUR_TOKEN`
- Manage documents and categories
- Configure AI models
- View usage analytics

## Technologies

- Streamlit
- LangChain
- FAISS Vector Store
- OpenRouter API
- HuggingFace Embeddings
- Plotly

## Requirements

- Python 3.8+
- See requirements.txt for full list

## License

MIT License

## Author

Ashish Kharbanda

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request 