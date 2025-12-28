# 📄 Papers Inbox

Place PDF versions of papers you read here.

## Folder Structure

Organize papers into subfolders by topic for easier browsing:
```
papers/
├── Causal Inference/
│   ├── Pearl_2009_Causality.pdf
│   └── Imbens_Rubin_2015_Causal_Inference.pdf
├── Research Methods/
│   └── Shadish_2002_Experimental_Design.pdf
├── Platform Design/
│   └── Kang_2024_Voice_Anonymization.pdf
└── README.md
```

Subfolder names appear in search results, helping you locate the source file.

---

## What Happens When You Add a Paper

1. **Text extraction** — PDF text is extracted page by page
2. **Metadata lookup** — Title, authors, and year are fetched via Crossref (when possible)
3. **Chunking** — Long papers are split into searchable segments
4. **Embedding** — Each chunk is converted to a vector for semantic search
5. **Summarization** — An AI summary is generated for the full paper

---

## Naming Conventions (Optional)

While not required, consistent naming helps you find files:
```
AuthorLastName_Year_ShortTitle.pdf
```

Examples:
- `Pearl_2009_Causality.pdf`
- `Vaswani_2017_Attention.pdf`
- `Mercier_Sperber_2011_Argumentative_Theory.pdf`

---

## Metadata Extraction

The system attempts to automatically extract:
- **Title** — From PDF metadata or Crossref lookup
- **Authors** — From PDF metadata or Crossref lookup
- **Year** — From PDF metadata or Crossref lookup

If automatic extraction fails, the filename and folder are used as fallbacks.

---

## Tips

- ✅ **Use descriptive folder names** — They appear in search results
- ✅ **Keep original PDFs** — Don't rename if metadata extraction works
- ✅ **Add papers you actually read** — The system indexes everything
- ✅ **Pair with notes** — Add your notes in `/notes/` for the same papers

- ❌ Don't add scanned PDFs without OCR (text can't be extracted)
- ❌ Don't add password-protected PDFs
- ❌ Don't add huge files (>50MB) without expecting slower processing

---

## Supported File Types

Currently supported:
- `.pdf` (PDF documents)

The system extracts text from PDFs. Scanned documents need OCR preprocessing.

---

## Getting Started

1. Create a subfolder for your topic area
2. Add your PDF files
3. Run the indexing pipeline: `python scripts/update_and_deploy.py`
4. Search your papers in the app!

---

## Finding Open-Access Papers

Need papers to test with? Try these sources:

- **arXiv** (arxiv.org) — Preprints in physics, math, CS, and more
- **PubMed Central** (ncbi.nlm.nih.gov/pmc) — Biomedical and life sciences
- **SSRN** (ssrn.com) — Social sciences and humanities
- **Semantic Scholar** (semanticscholar.org) — Multi-discipline with open access filter
- **Google Scholar** — Filter by "All versions" to find free PDFs