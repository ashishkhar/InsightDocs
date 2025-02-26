import streamlit as st
import os
import hashlib
import json
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from datetime import datetime
import pandas as pd
from openai import OpenAI
from langchain.llms.base import LLM
from typing import Any, List, Optional, Dict
import plotly.express as px

# Page config
st.set_page_config(page_title="InsightDocs", page_icon="📚")

# Initialize session states
if 'vector_store_ready' not in st.session_state:
    st.session_state.vector_store_ready = False
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False
if 'is_configured' not in st.session_state:
    st.session_state.is_configured = False
if 'uploaded_files_info' not in st.session_state:
    st.session_state.uploaded_files_info = []
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'model_config' not in st.session_state:
    st.session_state.model_config = {
        'model': 'deepseek/deepseek-r1-distill-llama-8b',
        'tier': 'free',
        'daily_limit': 100,  # messages per day
        'messages_today': 0,
        'last_reset_date': datetime.now().strftime('%Y-%m-%d')
    }
if 'token_usage' not in st.session_state:
    st.session_state.token_usage = {
        'total_tokens': 0,
        'total_cost': 0,
        'usage_history': [],
        'model_usage': {}
    }

# Config file path
CONFIG_FILE = "admin_config.json"

# Add OpenRouter configuration
OPENROUTER_API_KEY = "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"
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://insightdocs.ai",  # Replace with your site URL
    "X-Title": "InsightDocs",  # Your site name
}

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

class OpenRouterLLM(LLM):
    model: str = "deepseek/deepseek-r1-distill-llama-8b"
    temperature: float = 0.1
    client: Any = None
    api_key: str = OPENROUTER_API_KEY  # Add API key as class attribute
    
    def __init__(self, model="deepseek/deepseek-r1-distill-llama-8b", temperature=0.1, api_key=None):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.api_key = api_key if api_key else OPENROUTER_API_KEY
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
    
    @property
    def _llm_type(self) -> str:
        return "openrouter"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        try:
            completion = self.client.chat.completions.create(
                extra_headers=OPENROUTER_HEADERS,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            
            # Update token usage
            response_tokens = len(completion.choices[0].message.content.split())  # Approximate token count
            update_token_usage(response_tokens, self.model)
            save_token_usage()
            
            return completion.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenRouter API error: {str(e)}")
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "api_key": f"...{self.api_key[-4:]}"  # Only show last 4 chars of API key
        }

def load_admin_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            return config
        return None
    
def save_admin_config(username, password_hash, token):
    config = {
        "username": username,
        "password_hash": password_hash,
        "token": token
    }
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f)

def initial_setup():
    st.title("🔧 Initial Admin Setup")
    st.write("Welcome! Please set up your admin credentials.")
    
    with st.form("setup_form"):
        username = st.text_input("Choose Admin Username")
        password = st.text_input("Choose Admin Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        token = st.text_input("Choose Admin Access Token (for URL access)")
        
        submitted = st.form_submit_button("Create Admin Account")
        
        if submitted:
            if not username or not password or not token:
                st.error("All fields are required!")
                return False
            if password != confirm_password:
                st.error("Passwords don't match!")
                return False
            if len(password) < 8:
                st.error("Password must be at least 8 characters long!")
                return False
            if len(token) < 8:
                st.error("Access token must be at least 8 characters long!")
                return False
            
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            save_admin_config(username, password_hash, token)
            st.success("Admin account created successfully!")
            st.info(f"Your admin URL will be: http://your-app-url/?access_token={token}")
            st.warning("Please save your access token somewhere safe!")
            return True
    return False

def check_password():
    """Returns `True` if the user had the correct password."""
    config = load_admin_config()
    if not config:
        return False

    def password_entered():
        if (hashlib.sha256(st.session_state["password"].encode()).hexdigest() == config["password_hash"] and 
            st.session_state["username"] == config["username"]):
            st.session_state.is_admin = True
            st.session_state.password_correct = True
        else:
            st.session_state.is_admin = False
            st.session_state.password_correct = False

    if "password_correct" not in st.session_state:
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.button("Login", on_click=password_entered)
        return False
    elif not st.session_state.password_correct:
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.button("Login", on_click=password_entered)
        st.error("😕 Invalid username or password")
        return False
    else:
        return True

def process_pdfs(uploaded_files):
    if not uploaded_files:
        return None
    
    all_docs = []
    with st.spinner('Processing documents...'):
        try:
            for uploaded_file in uploaded_files:
                # Save uploaded file temporarily
                with open(f"temp_{uploaded_file.name}", "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # Load and process the PDF
                loader = PDFPlumberLoader(f"temp_{uploaded_file.name}")
                docs = loader.load()
                all_docs.extend(docs)
                
                # Clean up temporary file
                os.remove(f"temp_{uploaded_file.name}")
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            documents = text_splitter.split_documents(all_docs)
            
            embedder = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            vector_store = FAISS.from_documents(documents, embedder)
            
            return vector_store
            
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
            return None

def load_vector_store():
    try:
        if os.path.exists("vector_store"):
            embedder = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            vector_store = FAISS.load_local("vector_store", embedder)
            return vector_store
        return None
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None

def save_vector_store(vector_store):
    try:
        vector_store.save_local("vector_store")
        return True
    except Exception as e:
        st.error(f"Error saving vector store: {str(e)}")
        return False

def save_files_info(files_info):
    with open('files_info.json', 'w') as f:
        json.dump(files_info, f)

def load_files_info():
    try:
        with open('files_info.json', 'r') as f:
            return json.load(f)
    except:
        return []

def save_model_config():
    with open('model_config.json', 'w') as f:
        json.dump(st.session_state.model_config, f)

def load_model_config():
    try:
        with open('model_config.json', 'r') as f:
            return json.load(f)
    except:
        return st.session_state.model_config

def update_token_usage(response_tokens, model_name):
    """Update token usage statistics"""
    # Approximate cost per 1K tokens (adjust based on actual pricing)
    cost_per_1k = {
        "deepseek/deepseek-r1-distill-llama-8b": 0.0015,
        "meta-llama/llama-2-70b-chat": 0.005,
        "anthropic/claude-2": 0.008,
        "google/gemini-pro": 0.004
    }
    
    cost = (response_tokens / 1000) * cost_per_1k.get(model_name, 0.002)
    
    # Update total tokens and cost
    st.session_state.token_usage['total_tokens'] += response_tokens
    st.session_state.token_usage['total_cost'] += cost
    
    # Update model-specific usage
    if model_name not in st.session_state.token_usage['model_usage']:
        st.session_state.token_usage['model_usage'][model_name] = {
            'tokens': 0,
            'cost': 0,
            'calls': 0
        }
    
    st.session_state.token_usage['model_usage'][model_name]['tokens'] += response_tokens
    st.session_state.token_usage['model_usage'][model_name]['cost'] += cost
    st.session_state.token_usage['model_usage'][model_name]['calls'] += 1
    
    # Add to usage history
    st.session_state.token_usage['usage_history'].append({
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'tokens': response_tokens,
        'cost': cost,
        'model': model_name
    })

def save_token_usage():
    """Save token usage to file"""
    with open('token_usage.json', 'w') as f:
        json.dump(st.session_state.token_usage, f)

def load_token_usage():
    """Load token usage from file"""
    try:
        with open('token_usage.json', 'r') as f:
            return json.load(f)
    except:
        return st.session_state.token_usage

def admin_portal():
    st.title("📚 InsightDocs")
    st.write("Your Intelligent Document Assistant")
    
    # Create tabs for different admin functions
    tab1, tab2, tab3, tab4 = st.tabs(["Upload Documents", "Manage Documents", "Model Settings", "Advanced Analytics"])
    
    with tab1:
        st.write("Upload new documents")
        
        # File uploader
        uploaded_files = st.file_uploader("Upload PDF documents", type=['pdf'], accept_multiple_files=True)

        if uploaded_files:
            # Get existing files info
            files_info = load_files_info()
            
            # Process new files
            with st.form("file_details"):
                st.write("Enter details for uploaded files:")
                file_details = []
                
                for uploaded_file in uploaded_files:
                    st.subheader(f"File: {uploaded_file.name}")
                    display_name = st.text_input(
                        "Display Name",
                        value=uploaded_file.name.replace('.pdf', ''),
                        key=f"name_{uploaded_file.name}"
                    )
                    description = st.text_area(
                        "Description",
                        key=f"desc_{uploaded_file.name}"
                    )
                    category = st.selectbox(
                        "Category",
                        ["Manual", "Datasheet", "Guide", "Other"],
                        key=f"cat_{uploaded_file.name}"
                    )
                    file_details.append({
                        "original_name": uploaded_file.name,
                        "display_name": display_name,
                        "description": description,
                        "category": category,
                        "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "file_size": uploaded_file.size
                    })
                
                if st.form_submit_button("Process Files"):
                    vector_store = process_pdfs(uploaded_files)
                    if vector_store is not None:
                        if save_vector_store(vector_store):
                            # Update files info
                            files_info.extend(file_details)
                            save_files_info(files_info)
                            
                            st.success("Documents processed and saved successfully!")
                            st.session_state.vector_store_ready = True
                            st.session_state.vector_store = vector_store
                            st.experimental_rerun()
    
    with tab2:
        st.write("Manage existing documents")
        
        files_info = load_files_info()
        if not files_info:
            st.info("No documents uploaded yet.")
            return
        
        # Convert to DataFrame for better display
        df = pd.DataFrame(files_info)
        
        # Add management options
        for idx, file in enumerate(files_info):
            with st.expander(f"📄 {file['display_name']}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Category:** {file['category']}")
                    st.write(f"**Upload Date:** {file['upload_date']}")
                    st.write(f"**File Size:** {file['file_size']/1024:.1f} KB")
                    st.write(f"**Description:** {file['description']}")
                
                with col2:
                    # Edit file details
                    if st.button("Edit", key=f"edit_{idx}"):
                        st.session_state[f'editing_{idx}'] = True
                    
                    # Delete file
                    if st.button("Delete", key=f"delete_{idx}"):
                        if st.warning("Are you sure you want to delete this file?"):
                            files_info.pop(idx)
                            save_files_info(files_info)
                            st.success("File deleted successfully!")
                            st.experimental_rerun()
                
                # Show edit form if editing
                if st.session_state.get(f'editing_{idx}', False):
                    with st.form(f"edit_form_{idx}"):
                        new_name = st.text_input("Display Name", value=file['display_name'])
                        new_desc = st.text_area("Description", value=file['description'])
                        new_cat = st.selectbox(
                            "Category",
                            ["Manual", "Datasheet", "Guide", "Other"],
                            index=["Manual", "Datasheet", "Guide", "Other"].index(file['category'])
                        )
                        
                        if st.form_submit_button("Save Changes"):
                            files_info[idx].update({
                                "display_name": new_name,
                                "description": new_desc,
                                "category": new_cat
                            })
                            save_files_info(files_info)
                            st.session_state[f'editing_{idx}'] = False
                            st.success("Changes saved successfully!")
                            st.experimental_rerun()
        
        # Document statistics
        st.subheader("📊 Document Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Documents", len(files_info))
        with col2:
            st.metric("Categories", len(df['category'].unique()))
        with col3:
            st.metric("Total Size", f"{df['file_size'].sum()/1024/1024:.1f} MB")
        
        # Category distribution
        st.subheader("📈 Category Distribution")
        cat_counts = df['category'].value_counts()
        st.bar_chart(cat_counts)
        
        if st.button("Clear All Documents"):
            if st.warning("⚠️ Are you sure? This will delete all documents!"):
                if os.path.exists("vector_store"):
                    try:
                        import shutil
                        shutil.rmtree("vector_store")
                        save_files_info([])  # Clear files info
                        st.session_state.vector_store_ready = False
                        st.success("All documents cleared successfully!")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error clearing documents: {str(e)}")

    with tab3:
        st.subheader("🤖 AI Model Configuration")
        
        # Load latest config at the start of the tab
        current_config = load_model_config()
        
        # Model selection
        model_tier = st.radio(
            "Select Usage Tier",
            ["Free (Limited)", "Custom API Key (Unlimited)"],
            index=0 if current_config['tier'] == 'free' else 1
        )

        if model_tier == "Free (Limited)":
            st.info(f"Using default model: {current_config['model']}")
            st.warning(f"Limited to {current_config['daily_limit']} messages per day")
            
            # Show usage statistics from current config
            messages_today = current_config['messages_today']
            remaining = current_config['daily_limit'] - messages_today
            
            # Create columns for metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Messages Used Today", messages_today)
            with col2:
                st.metric("Messages Remaining", remaining)
            
            # Reset counter if it's a new day
            today = datetime.now().strftime('%Y-%m-%d')
            if today != current_config['last_reset_date']:
                current_config.update({
                    'messages_today': 0,
                    'last_reset_date': today
                })
                save_model_config()
                st.experimental_rerun()

        else:
            api_key = st.text_input(
                "Enter Your OpenRouter API Key",
                type="password",
                value=st.session_state.api_key if st.session_state.api_key else ""
            )
            
            available_models = [
                "deepseek/deepseek-r1-distill-llama-8b",
                "meta-llama/llama-2-70b-chat",
                "anthropic/claude-2",
                "google/gemini-pro"
            ]
            
            selected_model = st.selectbox(
                "Select Model",
                available_models,
                index=available_models.index(current_config['model'])
            )
            
            if st.button("Save Configuration"):
                if not api_key:
                    st.error("Please enter an API key")
                else:
                    # Test API key
                    try:
                        test_client = OpenAI(
                            base_url="https://openrouter.ai/api/v1",
                            api_key=api_key
                        )
                        test_client.chat.completions.create(
                            extra_headers=OPENROUTER_HEADERS,
                            model=selected_model,
                            messages=[{"role": "user", "content": "test"}],
                        )
                        
                        st.session_state.api_key = api_key
                        current_config.update({
                            'model': selected_model,
                            'tier': 'custom',
                            'daily_limit': float('inf')
                        })
                        save_model_config()
                        st.success("API key validated and configuration saved!")
                        
                    except Exception as e:
                        st.error(f"Invalid API key or model selection: {str(e)}")

    with tab4:
        st.subheader("📊 Advanced Analytics")
        
        # Load latest usage data
        usage_data = load_token_usage()
        
        # Overview metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Tokens Used", f"{usage_data['total_tokens']:,}")
        with col2:
            st.metric("Total Cost", f"${usage_data['total_cost']:.2f}")
        
        # Model usage breakdown
        st.subheader("Model Usage Breakdown")
        if usage_data['model_usage']:
            model_df = pd.DataFrame([
                {
                    'Model': model.split('/')[-1].replace('-', ' ').title(),  # Clean model names
                    'Tokens': stats['tokens'],
                    'Cost': stats['cost']
                }
                for model, stats in usage_data['model_usage'].items()
            ])
            
            # Format cost for display
            model_df['Display Cost'] = model_df['Cost'].apply(lambda x: f"${x:.3f}")
            st.dataframe(
                model_df[['Model', 'Tokens', 'Display Cost']],
                hide_index=True,
                column_config={
                    "Tokens": st.column_config.NumberColumn(format="%d"),
                    "Display Cost": st.column_config.TextColumn("Cost")
                }
            )
            
            # Usage visualization
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Token Usage Distribution")
                fig_tokens = px.pie(
                    model_df, 
                    values='Tokens',
                    names='Model',
                    title="Token Distribution by Model"
                )
                fig_tokens.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    hole=.4,  # Make it a donut chart
                    pull=[0.1] * len(model_df)  # Slight separation between segments
                )
                fig_tokens.update_layout(
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig_tokens, use_container_width=True)
            
            with col2:
                st.subheader("Cost Distribution")
                fig_cost = px.pie(
                    model_df, 
                    values='Cost',
                    names='Model',
                    title="Cost Distribution by Model"
                )
                fig_cost.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    hole=.4,  # Make it a donut chart
                    pull=[0.1] * len(model_df)  # Slight separation between segments
                )
                fig_cost.update_layout(
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig_cost, use_container_width=True)

        # Usage history
        st.subheader("Daily Usage Trends")
        if usage_data['usage_history']:
            history_df = pd.DataFrame(usage_data['usage_history'])
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            history_df['date'] = history_df['timestamp'].dt.date
            
            # Daily usage trend with bar charts
            daily_usage = history_df.groupby('date').agg({
                'tokens': 'sum',
                'cost': 'sum'
            }).reset_index()
            
            # Create two separate bar charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Daily Token Usage")
                fig_tokens = px.bar(
                    daily_usage,
                    x='date',
                    y='tokens',
                    title="Tokens per Day"
                )
                fig_tokens.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Tokens",
                    bargap=0.2,
                    height=400,
                    xaxis=dict(
                        tickangle=45,
                        tickformat="%Y-%m-%d"
                    )
                )
                fig_tokens.update_traces(
                    marker_color='rgb(55, 83, 109)'
                )
                st.plotly_chart(fig_tokens, use_container_width=True)
            
            with col2:
                st.subheader("Daily Cost")
                fig_cost = px.bar(
                    daily_usage,
                    x='date',
                    y='cost',
                    title="Cost per Day"
                )
                fig_cost.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Cost ($)",
                    bargap=0.2,
                    height=400,
                    xaxis=dict(
                        tickangle=45,
                        tickformat="%Y-%m-%d"
                    )
                )
                fig_cost.update_traces(
                    marker_color='rgb(26, 118, 255)'
                )
                st.plotly_chart(fig_cost, use_container_width=True)
        
        # Export data
        if st.button("Export Usage Data"):
            # Prepare export data
            export_df = pd.DataFrame(usage_data['usage_history'])
            export_df['timestamp'] = pd.to_datetime(export_df['timestamp'])
            export_df['cost'] = export_df['cost'].apply(lambda x: f"${x:.3f}")
            
            csv = export_df.to_csv(index=False)
            st.download_button(
                "Download Usage History",
                csv,
                "token_usage.csv",
                "text/csv",
                key='download-csv'
            )
        
        # Reset statistics
        if st.button("Reset Usage Statistics"):
            if st.warning("⚠️ Are you sure? This will delete all usage statistics!"):
                st.session_state.token_usage = {
                    'total_tokens': 0,
                    'total_cost': 0,
                    'usage_history': [],
                    'model_usage': {}
                }
                save_token_usage()
                st.success("Usage statistics reset successfully!")
                st.experimental_rerun()

def user_portal():
    st.title("📚 InsightDocs")
    st.write("Ask questions related to company's knowledge base")

    # Load vector store if not already loaded
    if not st.session_state.vector_store_ready:
        vector_store = load_vector_store()
        if vector_store is not None:
            st.session_state.vector_store = vector_store
            st.session_state.vector_store_ready = True

    if not st.session_state.vector_store_ready:
        st.info("No documents available. Please contact the administrator.")
        return

    # Check and update daily limit
    today = datetime.now().strftime('%Y-%m-%d')
    if today != st.session_state.model_config['last_reset_date']:
        st.session_state.model_config.update({
            'messages_today': 0,
            'last_reset_date': today
        })
        save_model_config()
    
    # Load latest config
    current_config = load_model_config()
    st.session_state.model_config = current_config

    if st.session_state.model_config['tier'] == 'free':
        if st.session_state.model_config['messages_today'] >= st.session_state.model_config['daily_limit']:
            st.error("Daily message limit reached. Please try again tomorrow or contact administrator for unlimited access.")
            return

    # Initialize OpenRouter LLM
    try:
        api_key = st.session_state.api_key if st.session_state.model_config['tier'] == 'custom' else OPENROUTER_API_KEY
        llm = OpenRouterLLM(
            model=st.session_state.model_config['model'],
            temperature=0.1,
            api_key=api_key
        )
    except Exception as e:
        st.error(f"Error connecting to AI model: {str(e)}")
        return
    
    # Set up QA chain
    prompt = """You are a helpful assistant for answering questions about company documents. Use the following context to answer questions 
    accurately and professionally. If you're not sure about something, please say so.
    
    Context: {context}
    Question: {question}
    
    Think deeply about the question and context, then provide your answer. Put your thinking process between <think> and </think> tags, and your final answer after that.
    
    Answer: Let me help you with that.
    <think>
    Let me analyze the information from the documents and provide a clear answer...
    </think>"""
    
    def extract_answer(text):
        if "</think>" in text:
            return text.split("</think>")[-1].strip()
        return text
    
    QA_PROMPT = PromptTemplate(
        template=prompt,
        input_variables=["context", "question"]
    )
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=st.session_state.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        ),
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True
    )
    
    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask about documents:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            try:
                input_tokens = len(prompt.split())
                response = qa({"query": prompt})
                answer = extract_answer(response["result"])
                st.markdown(answer)
                
                response_tokens = len(answer.split())
                total_tokens = input_tokens + response_tokens
                update_token_usage(total_tokens, st.session_state.model_config['model'])
                save_token_usage()
                
                with st.expander("View Sources"):
                    for doc in response["source_documents"]:
                        st.markdown(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                        st.markdown(f"**Excerpt:** {doc.page_content[:200]}...")
                
                st.session_state.messages.append({"role": "assistant", "content": answer})

                # Update message counter after successful response
                if st.session_state.model_config['tier'] == 'free':
                    st.session_state.model_config['messages_today'] += 1
                    save_model_config()
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.experimental_rerun()

# Main app logic
def main():
    config = load_admin_config()
    
    # Show initial setup if no config exists
    if not config:
        if initial_setup():
            st.button("Continue to App", on_click=lambda: st.experimental_rerun())
        return

    # Check for admin access via URL parameter
    is_admin_view = st.query_params.get("access_token", "") == config["token"]

    if is_admin_view:
        if check_password():
            admin_portal()
            if st.button("Exit Admin Mode"):
                st.query_params.clear()
                st.session_state.clear()
                st.experimental_rerun()
        else:
            st.title("Admin Login")
    else:
        user_portal()

if __name__ == "__main__":
    main()
