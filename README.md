<div align="center">

# ZenShine Pro
### AI-Powered RAG Customer Support Assistant

<p align="center">
  <img src="https://img.shields.io/badge/LLM-Groq-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/RAG-Qdrant-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Frontend-Streamlit-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge" />
</p>

<p align="center">
  Enterprise-grade AI chatbot with semantic search, PDF ingestion, multilingual support, and premium UI/UX.
</p>

</div>

---

# Overview

ZenShine Pro is a production-ready AI customer support assistant designed for service businesses.

This project combines:

- Retrieval-Augmented Generation (RAG)
- Semantic search
- Vector databases
- Large Language Models
- Premium frontend engineering
- Intelligent pricing workflows

The chatbot can:

- Answer customer queries
- Retrieve information from uploaded PDFs
- Generate context-aware responses
- Handle multilingual conversations
- Calculate discounts automatically
- Recommend service packages

---

# Features

## AI-Powered Conversational Assistant

- Context-aware responses
- Natural conversations
- Dynamic customer interaction
- Smart service recommendations

---

## RAG Architecture

Retrieval-Augmented Generation pipeline using:

- Qdrant Vector Database
- SentenceTransformers embeddings
- Semantic retrieval
- Context injection into LLM

---

## Semantic Search

Supports intelligent document retrieval using:

- all-MiniLM-L6-v2 embeddings
- Cosine similarity search
- Vector indexing
- Top-K retrieval strategy

---

## PDF Knowledge Base

Users can upload PDFs directly into the application.

The system automatically:

1. Extracts text
2. Chunks content
3. Generates embeddings
4. Stores vectors in Qdrant
5. Enables semantic querying

---

## Multilingual Support

Supports:

- English
- Tamil
- Hindi

The chatbot automatically replies in the user's language.

---

## Premium UI/UX

Custom-built Streamlit interface featuring:

- Enterprise-style layout
- Animated status indicators
- Modern sidebar design
- Responsive chat interface
- Premium typography
- Custom CSS architecture

---

# Tech Stack

| Technology | Usage |
|---|---|
| Python | Backend |
| Streamlit | Frontend UI |
| Groq | LLM Inference |
| Qdrant | Vector Database |
| SentenceTransformers | Embeddings |
| PyMuPDF | PDF Processing |
| dotenv | Environment Variables |
| httpx | HTTP Client |

---

# System Architecture

```text
User Query
    ↓
Streamlit Frontend
    ↓
Embedding Generation
    ↓
Qdrant Vector Search
    ↓
Top-K Context Retrieval
    ↓
Groq LLM
    ↓
AI Generated Response
Project Structure
Chatbot-Demo/
│
├── app.py
├── rag.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── data/
    └── ChatBot_Q_A's.pdf
Core AI Components
Retrieval Pipeline
Semantic chunk retrieval
Score threshold filtering
Payload indexing
Context injection
Embedding Model
all-MiniLM-L6-v2

Used for semantic vector generation.

LLM Model
llama-3.3-70b-versatile

Served using Groq inference.

Installation
Clone Repository
git clone https://github.com/manobhisriram/Chatbot-Demo.git
cd Chatbot-Demo
Create Virtual Environment
Windows
python -m venv venv
venv\Scripts\activate
Mac/Linux
python3 -m venv venv
source venv/bin/activate
Install Dependencies
pip install -r requirements.txt
Configure Environment Variables

Create a .env file:

GROQ_API_KEY=your_groq_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
Run Application
streamlit run app.py
Sample Queries
• What services do you provide?
• What is included in full house cleaning?
• Do you offer combo discounts?
• House + kitchen + bathroom cleaning total?
• How can I book a service?
Engineering Highlights
Prompt Engineering
Business-specific system prompts
Controlled AI responses
Dynamic pricing logic
Conversational alignment
Production RAG Design
Context-aware retrieval
Efficient vector search
Scalable architecture
Modular AI pipeline
Enterprise Frontend
Custom CSS styling
Dynamic layouts
Interactive chat UI
Responsive components
Future Improvements
WhatsApp AI Agent
Voice AI Integration
LangChain / LangGraph
Multi-agent workflows
CRM integration
AI analytics dashboard
Autonomous booking workflows
Resume Impact

This project demonstrates:

Generative AI Engineering
RAG System Design
Vector Database Integration
Semantic Search
LLM Application Development
AI Product Engineering
Production-grade Frontend Development
Author
Manu

Generative AI Engineer | Agentic AI Engineer

Building:

AI Agents
RAG Applications
LLM Products
Enterprise AI Systems
AI Workflow Automation
GitHub Topics
generative-ai
rag
llm
streamlit
qdrant
groq
semantic-search
vector-database
python
ai-chatbot
Star This Repository

If you found this project useful, give it a star.
