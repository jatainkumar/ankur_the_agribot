### AgriBot Project - Google Colab Setup and Usage Guide

#### Overview
This notebook implements **AgriBot**, an AI-powered agricultural assistant designed for Indian farmers. It leverages a Retrieval-Augmented Generation (RAG) architecture, combining a knowledge base of agricultural PDFs with real-time data from external APIs. The bot provides actionable advice on irrigation, pest control, market prices, and weather, with support for both English and Hindi queries.

***

### Step 1: Initial Setup

#### 1.1 Mount Google Drive
To store and access your PDF documents and the search index, you need to mount your Google Drive.

```python
from google.colab import drive
drive.mount('/content/drive')
```

#### 1.2 Install Required Packages
The project depends on several Python libraries. Run the following commands in a Colab cell to install them.

```python
!pip install --no-cache-dir \
 pymupdf pdfplumber tabula-py faiss-cpu rank-bm25 unidecode rapidfuzz \
 pydantic==2.7.1 transformers==4.30.0 \
 sentence-transformers==2.2.0 \
 huggingface-hub==0.23.0

!pip install huggingface-hub==0.25.2
!pip install -U sentence-transformers
!pip install gradio
!pip install gtts
!pip install speechrecognition
!pip install groq
```

***

### Step 2: Configuration

#### 2.1 Set Up Folder Paths
Before running the main logic, you must define the paths for your PDF knowledge base and the search index files. Create these folders in your Google Drive and update the paths in the notebook.

```python
import os

# Define the directory where your agricultural PDF files are stored
PDF_DIR = "/content/drive/MyDrive/AgriBot_Project_3/content/pdfs"

# Define the directory where the search index will be saved and loaded from
INDEX_DIR = "/content/drive/MyDrive/AgriBot_Project_3/content/index"

# Create the directories if they don't already exist
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)
```

#### 2.2 Configure API Keys
The bot uses several external services to fetch real-time data. You need to obtain API keys for these services and set them as environment variables in your Colab session.

```python
import os

# Replace the placeholder text with your actual API keys
os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"
os.environ["OPENWEATHER_API_KEY"] = "your_openweather_api_key_here"
os.environ["TAVILY_API_KEY"] = "your_tavily_api_key_here"
os.environ["AGROMONITORING_API_KEY"] = "your_agromonitoring_api_key_here"
os.environ["DATA_GOV_IN_API_KEY"] = "your_data_gov_in_api_key_here"
```

**How to get the API Keys**:
- **Groq**: Sign up on the [Groq platform](https://console.groq.com/keys) to get your API key.
- **OpenWeather**: Create a free account at [openweathermap.org](https://openweathermap.org) to get an API key.
- **Tavily**: Register at [tavily.com](https://tavily.com) for API access.
- **AgroMonitoring**: Sign up for an API key at [agromonitoring.com](https://agromonitoring.com).
- **Data.gov.in**: Register on the [data.gov.in](https://data.gov.in) portal to get an API key.

***

### Step 3: Data Ingestion and Index Building
This step processes your PDFs and builds a searchable index.

#### 3.1 Upload PDFs
Place all your agricultural PDF documents into the folder specified by `PDF_DIR`. The notebook will automatically scan this directory.

#### 3.2 Run the Ingestion Script
To create the knowledge base, you need to uncomment and run the ingestion cell in the notebook. This process extracts text and tables from each PDF, chunks the content, and builds a hybrid search index using FAISS (for dense retrieval) and BM25 (for sparse retrieval).

```python
# This code snippet is present in the notebook but commented out.
# You must uncomment it to build your index for the first time.

hybrid = HybridIndex(embedding_model_name=EMBEDDING_MODEL_NAME)
pdf_files = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))

total_chunks = 0
for pdf_path in pdf_files:
    meta = default_doc_meta(pdf_path)
    try:
        t_chunks = extract_text_chunks(pdf_path, meta)
        tb_chunks = extract_tables(pdf_path, meta)
        hybrid.add_chunks(t_chunks + tb_chunks)
        total_chunks += len(t_chunks) + len(tb_chunks)
        print(f"Ingested {os.path.basename(pdf_path)}: text={len(t_chunks)}, table_chunks={len(tb_chunks)}")
    except Exception as e:
        print("Failed:", pdf_path, e)

print("Total chunks:", total_chunks)
hybrid.build()
hybrid.save(INDEX_DIR)
print("Index built and saved to", INDEX_DIR)
```

***

### Step 4: Launch the AgriBot User Interface
After building the index, you can launch the interactive Gradio interface to chat with the bot.

#### 4.1 Load the Index
First, ensure the index is loaded into memory.

```python
hybrid2 = HybridIndex(embedding_model_name=EMBEDDING_MODEL_NAME)
hybrid2.load(INDEX_DIR)
```

#### 4.2 Run the Gradio App
Execute the final cell in the notebook to start the Gradio web interface.

```python
demo.launch(debug=True, share=True)
```
This will provide a public URL that you can open in your browser to interact with AgriBot. The interface supports both text and voice input.

### Features of the UI:
- **Text Chat**: Type your agricultural questions directly.
- **Voice Input**: Record your questions using the microphone.
- **Audio Response**: Listen to the bot's answers.

***

### Notes and Troubleshooting
- **File Paths**: Ensure that the paths to your PDF and index directories are correct. Incorrect paths are a common source of errors.
- **API Quotas**: Be mindful of the usage limits for the free tiers of the APIs. If you encounter errors, check your API dashboards for quota information.
- **Dependencies**: If you face issues with package installations, try restarting the Colab runtime (`Runtime > Restart session`) and running the installation cells again.
- **Language Support**: The bot is designed to handle both English and Hindi queries, making it accessible to a wider range of farmers in India.
