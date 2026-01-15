# 🧠 Pensieve

A hybrid semantic search tool for academic papers and research notes, powered by ChromaDB and OpenAI embeddings.

Just like Dumbledore stored memories in the Pensieve, this tool helps you retrieve knowledge from your notes and readings - making your research instantly searchable and intelligently summarized.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pensieve-live.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Attribution

If you use or adapt this project, please credit Rehan Mirza as the original author.
Academic citation is appreciated but not required beyond the MIT License.

---

## ✨ Features

- **Hybrid Search:** Find relevant content using semantic similarity, keyword matching (BM25), and metadata boosting — author name searches and specific concepts rank higher
- **Auto-Summarization**: LLM-generated summaries for each paper and note section
- **AI Query Snippets**: On-demand, query-focused insights that explain how each result relates to your search
- **Incremental Indexing**: Only processes new or changed files—fast updates
- **Dual Deployment**: Run locally or deploy to the cloud via Backblaze B2 + Streamlit Cloud

---

## 🚀 Quick Start (Local Mode)

Get up and running in 5 minutes:

```bash
# 1. Clone the repository
git clone https://github.com/ramirza1/pensieve-project-template.git
cd pensieve-project-template

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY (remove or comment out the B2 lines)
```

Your `.env` file should look like this for local use:
```env
OPENAI_API_KEY=sk-your-openai-api-key

# Comment out B2 settings for local-only use:
# B2_KEY_ID=your-b2-key-id
# B2_APP_KEY=your-b2-application-key
# B2_BUCKET_NAME=your-bucket-name
```

> ⚠️ **Important:** Make sure the B2 lines are commented out (with `#`) or removed for local-only use. Otherwise the app will try to download from cloud storage.

```bash
# 5. Index your content (creates the local database)
python scripts/update_and_deploy.py --skip-upload

# 6. Run the app
streamlit run app/streamlit_app.py
```

The app will open at `http://localhost:8501`.

> **Note:** Costs depend on the number of documents processed, document length, and chunking settings. Please monitor your OpenAI usage carefully.

---

## 📁 Organizing Your Content

### Notes (`data/inbox/notes/`)

Save your research notes as `.docx` files with this heading structure:
```
data/inbox/notes/
├── Example Notes/
│   └── Sample_Research_Notes.docx    ← Check this for formatting examples!
├── Your Topic Area/
│   └── Your_Notes.docx
└── README.md                          ← Detailed formatting guide
```

**Within each notes file:**

| Heading Level | Purpose | Example |
|---------------|---------|---------|
| **Heading 2** | Topic/Theme | `## Causal Inference` |
| **Heading 3** | Paper reference | `### Paper Title; Authors; Year` |
| Body text | Your notes | Bullet points, summaries, quotes |

**Example:**
```
## Research Methods                              ← Heading 2

### "Experimental Design"; Shadish et al.; 2002  ← Heading 3
- Key points about threats to validity
- Notes on randomization

### "Causal Inference"; Pearl; 2009              ← Heading 3
- DAGs and counterfactuals
- Do-calculus basics
```

### Papers (`data/inbox/papers/`)

Drop PDF files into this folder:
```
data/inbox/papers/
├── Pearl_2009_Causality.pdf
├── Vaswani_2017_Attention.pdf
└── subfolder/
    └── More_Papers.pdf
```

The system automatically extracts metadata (title, authors, year) from PDFs via Crossref lookup.

> 📖 See `data/inbox/notes/README.md` and `data/inbox/papers/README.md` for detailed formatting guides.

---

## ⚙️ Configuration

All settings are in `config.yaml`:
```yaml
# Embedding model (OpenAI)
embedding_model: text-embedding-3-small

# Paths (relative to project root)
paths:
  notes_inbox: data/inbox/notes
  papers_inbox: data/inbox/papers
  processed: data/processed

# Retrieval settings
retrieval:
  top_k: 5

# App settings
app:
  query_snippet_model: gpt-4.1-mini
  github_url: https://github.com/yourusername/pensieve
  allow_debug: false  # Set to true to show debug toggle

# OpenAI settings (used by indexing scripts)
openai:
  summarization_model: gpt-4.1-mini
```

---

## 🔧 Running the Pipeline

After adding or updating files in `data/inbox/notes/` or `data/inbox/papers/`, run the indexing pipeline:

### Basic Commands

```bash
# Full pipeline (index + summarize) - most common command
python scripts/update_and_deploy.py --skip-upload

# Only process notes
python scripts/update_and_deploy.py --notes-only --skip-upload

# Only process papers
python scripts/update_and_deploy.py --papers-only --skip-upload
```

### Additional Options

```bash
# Preview what would happen without making changes
python scripts/update_and_deploy.py --dry-run

# Force full reprocess (ignore cache)
python scripts/update_and_deploy.py --force --skip-upload

# Clean up orphaned entries (after deleting source files)
python scripts/update_and_deploy.py --cleanup --skip-upload
```

### Command Reference

| Flag | Description |
|------|-------------|
| `--skip-upload` | Index locally only (no cloud upload) — **use this for local mode** |
| `--notes-only` | Only process notes (skip papers) |
| `--papers-only` | Only process papers (skip notes) |
| `--force` | Force full reprocess (ignore cache) |
| `--cleanup` | Remove orphaned database entries |
| `--dry-run` | Preview changes without executing |
| `--skip-index` | Skip indexing (only summarize) |
| `--skip-summarize` | Skip summarization (only index) |

---

## 🔍 Using the App

### Search

1. Enter a topic or question in the search bar
   - Example: `"causal inference"`, `"attention mechanism"`, `"experimental design"`
2. Toggle **📝 Notes** and/or **📄 Papers** to filter results
3. Adjust **Number of results** (1-25) as needed

### Understanding Results

| Field | Notes | Papers |
|-------|-------|--------|
| **Theme** | Your Heading 2 topic | — |
| **Title** | From Heading 3 | Extracted from PDF |
| **Authors** | From Heading 3 | Extracted from PDF |
| **Year** | From Heading 3 | Extracted from PDF |
| **Summary** | Auto-generated | Auto-generated |

### AI Query Snippets

Toggle **✨ AI snippets** to enable query-focused insights:

- Click **"Generate / refresh"** on any result
- Get a direct answer explaining relevance to your query
- See 3-6 specific bullet points of insights
- Confidence rating (High/Medium/Low)

### Controls

- **🌗** Toggle light/dark mode
- **🐛** Toggle debug mode (shows chunk IDs, distances, metadata) — *only visible if `app.allow_debug: true` in config.yaml*

---

## ☁️ Cloud Deployment (Optional)

Want to access your Pensieve from anywhere? Deploy to the cloud:

### Prerequisites

1. **Backblaze B2 account** (free tier available)
   - Create a bucket for your ChromaDB files
   - Generate application keys

2. **Streamlit Cloud account** (free)
   - Connect your GitHub repository

### Setup

1. **Add B2 credentials to `.env`:**
```env
B2_KEY_ID=your-key-id
B2_APP_KEY=your-key
B2_BUCKET_NAME=your-bucket
B2_PREFIX=chroma_db/
```

2. **Run the full pipeline (with upload):**
```bash
python scripts/update_and_deploy.py
```

3. **Deploy to Streamlit Cloud:**
   - Push your repo to GitHub
   - Connect to [share.streamlit.io](https://share.streamlit.io)
   - Add your secrets in Streamlit Cloud settings:
     ```toml
     OPENAI_API_KEY = "sk-your-key"
     B2_KEY_ID = "your-key-id"
     B2_APP_KEY = "your-app-key"
     B2_BUCKET_NAME = "your-bucket"
     B2_PREFIX = "chroma_db/"
     # Optional: force re-download of database
     # FORCE_B2_DOWNLOAD = "true"
     ```

### Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                     YOUR LOCAL MACHINE                          │
├─────────────────────────────────────────────────────────────────┤
│  data/inbox/notes/*.docx  ──┐                                   │
│  data/inbox/papers/*.pdf  ──┼──► Indexing Scripts ──► ChromaDB  │
│                             │         │                         │
│                             │         ▼                         │
│                             │    LLM Summaries                  │
│                             │         │                         │
│                             │         ▼                         │
│                             └──► Upload to B2                   │
└─────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                     BACKBLAZE B2                                │
│                 (Cloud ChromaDB Storage)                        │
└─────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                   STREAMLIT CLOUD                               │
│          Downloads DB from B2 → Serves App                      │
└─────────────────────────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **Vector Database** | ChromaDB |
| **Embeddings** | OpenAI `text-embedding-3-small` |
| **Summarization** | OpenAI `gpt-4.1-mini` |
| **Cloud Storage** | Backblaze B2 (optional) |
| **Hosting** | Streamlit Cloud (optional) |

---

## 📋 Requirements

- **Python 3.10+**
- **OpenAI API key** ([Get one here](https://platform.openai.com/api-keys))
- **~$0.01-0.05 per paper** for embeddings and summaries (varies by length)

---

## 🗂️ Project Structure
```
pensieve-project-template/
├── app/
│   └── streamlit_app.py          # Main Streamlit application
├── src/
│   ├── index_notes.py            # Notes indexing logic
│   ├── index_papers.py           # Papers indexing logic
│   ├── summarize_notes.py        # Notes summarization
│   ├── summarize_papers.py       # Papers summarization
│   ├── chunking.py               # Text chunking utilities
│   ├── registry.py               # File tracking
│   └── orphan_cleanup.py         # Database cleanup
├── scripts/
│   └── update_and_deploy.py      # Main pipeline script
├── data/
│   ├── inbox/
│   │   ├── notes/                # Your .docx notes go here
│   │   └── papers/               # Your .pdf papers go here
│   ├── processed/                # Cache files (auto-generated)
│   └── _local/                   # Local ChromaDB (auto-generated)
├── assets/
│   └── pensieve.jpg              # App logo
├── config.yaml                   # Configuration
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment template
└── README.md                     # You are here!
```

---

## 💡 Tips

- **Consistent formatting**: Use Heading 2/3 consistently in your notes
- **Paper titles in H3**: Format as `Title; Authors; Year` for best metadata extraction
- **Folder organization**: Group related notes/papers into folders—they appear in search results
- **Regular updates**: Run the pipeline after adding new content
- **Start with examples**: Check `data/inbox/notes/Example Notes/` for a working template

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

- 🐛 Report bugs
- 💡 Suggest features
- 🔧 Submit pull requests

---

## 📄 License

This project is licensed under the MIT License—see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Inspired by the Pensieve from Harry Potter
- Built with [Streamlit](https://streamlit.io), [ChromaDB](https://www.trychroma.com/), and [OpenAI](https://openai.com)

---