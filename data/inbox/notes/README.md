data/inbox/notes/README.mdmarkdown# 📝 Notes Inbox

Place your research notes here as `.docx` files.

## Folder Structure

Organize notes into subfolders by course, project, or topic area:notes/
├── Example Notes/
│   └── Sample_Research_Notes.docx    ← Start here!
├── Research Methods/
│   └── Causal_Inference_Notes.docx
├── My Thesis/
│   └── Literature_Review.docx
└── README.md

Subfolder names appear in search results, helping you identify where content came from.

---

## Formatting Requirements

Use this heading structure for best results:

| Heading Level | Purpose | Example |
|---------------|---------|---------|
| **Heading 2** | Topic/Theme | `## Causal Inference` |
| **Heading 3** | Paper reference | `### Paper Title; Author(s); Year` |
| Body text | Your notes | Bullet points, quotes, summaries |

### Example Structure Inside a .docx FileResearch Methods                              ← Heading 2 (Theme)"Experimental Design", Shadish et al., 2002  ← Heading 3 (Paper)

Key points about threats to validity
Notes on randomization
See p. 45 for diagram
"Causal Inference", Pearl, 2009              ← Heading 3 (Paper)

DAGs and counterfactuals
Do-calculus basics
External Validity                             ← Heading 2 (Theme)"Generalization", Mutz, 2011                 ← Heading 3 (Paper)

Notes on this paper...


---

## What Gets Indexed

| Your Content | Becomes |
|--------------|---------|
| Heading 2 text | **Theme** in search results |
| Heading 3 text | Parsed into **Title**, **Authors**, **Year** |
| Body text under H3 | Searchable content + AI summaries |

Each Heading 3 section is indexed separately, so you can search for specific papers within your notes.

---

## Heading 3 Format

For best metadata extraction, format paper headings as:Paper Title; Author(s); Year

Examples:
- `### Thinking, Fast and Slow; Kahneman; 2011`
- `### Attention Is All You Need; Vaswani et al.; 2017`
- `### The Argumentative Theory of Reasoning; Mercier and Sperber; 2011`

The system will parse these into structured metadata displayed in search results.

---

## Tips

- ✅ **Include URLs** to papers for easy reference
- ✅ **Use bullet points** for key takeaways
- ✅ **Add page numbers** for important quotes
- ✅ **Group related papers** under the same Heading 2 theme
- ✅ **Be consistent** with your Heading 3 format

- ❌ Don't put multiple papers under one Heading 3
- ❌ Don't skip Heading 2 (go straight to H3)
- ❌ Don't use Heading 1 (reserved for document title)

---

## Getting Started

1. Look at `Example Notes/Sample_Research_Notes.docx` for a working example
2. Create your own `.docx` file with Heading 2/3 structure
3. Run the indexing pipeline: `python scripts/update_and_deploy.py`
4. Search your notes in the app!

---

## Supported File Types

Currently supported:
- `.docx` (Microsoft Word)

Not yet supported:
- `.doc` (legacy Word)
- `.md` (Markdown)
- `.txt` (Plain text)