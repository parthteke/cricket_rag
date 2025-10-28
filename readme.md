# Cricket RAG Bot

A simple Cricket Q&A bot using **FAISS + SentenceTransformers**.

It answers cricket-related questions using a CSV database of cricket facts. For unrelated queries, it responds with:
> "I'm not able to answer that question."

## Files

- `cricket_rag_bot.py` — main Python script
- `cricket_rag.csv` — CSV file containing cricket facts
- `INSTALLATION.md` — instructions to install dependencies and run the bot

## Usage

1. Ensure `cricket_rag.csv` is in the same folder as `cricket_rag_bot.py`.
2. Follow the steps in `INSTALLATION.md` to set up the environment.
3. Run the bot:
```bash
python cricket_rag_bot.py
