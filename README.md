# TwilightPaper

## Project overview

- Agentic AI Assignment 5.
- Upload a research PDF, ask questions about it, create an editable PowerPoint, and practise for a viva.
- Inspect retrieved passages, PDF page citations, and the prompts used for each answer.

## Assignment requirements

### 1. LLM integration

- The default provider is **Ollama**, running **Qwen3 4B** locally. OpenAI, Google Gemini, and Groq are also selectable.
- **Show it:** Select **Live AI** and **Ollama**, then click **Test model connection**.
- **Code:** [Model configuration and client](paper_viva/models.py#L44)

### 2. Interactive GUI

- The Streamlit interface provides PDF upload and indexing, a chat input, and tabs for chat, slides, viva practice, and session history.
- **Show it:** Upload and index a PDF, type a question in **Paper chat**, and display the answer.
- **Code:** [Streamlit workspace](app.py#L201)

### 3. Prompt engineering

- The intent router uses **few-shot prompting**. Its system prompt defines the available intents. Example user requests and expected assistant outputs show how to classify them.
- The router's user prompt supplies the latest request and recent conversation. The paper agent's system prompt defines its role, evidence rules, and output format. Its user prompt supplies the selected PDF, conversation, and request.
- **Show it:** Type a request and open **Run details · prompts and retrieval** to show the labeled system and user prompts and few-shot examples.
- **Code:** [Router examples](paper_viva/agents.py#L29) and [paper agent prompt](paper_viva/agents.py#L81)

### 4. LangChain core components

- `ChatPromptTemplate` builds the prompt. The LCEL router chain connects the template, model, final answer parser, and Pydantic route parser with `|`.
- **Show it:** Open the router chain and compare its parsed result with **Detected intent** in the app.
- **Code:** [LCEL routing chain](paper_viva/agents.py#L188)

### 5. Agentic RAG

- The model can call `search_paper` with its own query. Retrieved passages return to the model as tool results. The app checks that cited PDF pages appeared in those results.
- If the model skips retrieval on a paper task, a fallback search supplies evidence and is labeled in **Run details**. App help takes a separate route with zero searches.
- **Show it:** Ask **How did the authors train DistilBERT using knowledge distillation?** Show the search query, passages, and page citations. Then ask **How do I use this app?** and show zero searches.
- **Code:** [Search tool and agent loop](paper_viva/agents.py#L218)

### 6. Intent detection and routing

- The router classifies typed requests and dispatches them to the summary, concept, slides, viva, help, or clarification handler. Shortcut buttons use the intent selected by the user.
- **Show it:** Type **Explain the authors’ method** and then **How do I use this app?** Open **Run details** on each response to compare the detected intents and handlers.
- **Code:** [Request routing](paper_viva/agents.py#L173)

### 7. Vector database

- ChromaDB stores PDF passage embeddings with page metadata and retrieves matching passages using cosine similarity. Semantic indexing uses MiniLM. Keyword search is also available.
- **Show it:** Select **Semantic** before indexing a PDF. Show the ChromaDB passage count in the sidebar and the retrieved excerpts with page numbers in **Run details**.
- **Code:** [Similarity query](paper_viva/paper.py#L45) and [vector indexing](paper_viva/paper.py#L103)

## Verification

- Ollama returned `READY` using the local Qwen3 4B model.
- The Streamlit test interface accepted a typed request and displayed a live response with **Run details**.
- Semantic indexing of the DistilBERT PDF stored **26 passages across 5 pages** in ChromaDB.
- A typed DistilBERT question routed to **Concept Extractor**. The model chose its own search query, retrieved passages, and returned PDF page citations. The fallback search was not used.
- A typed app-help question routed to **App Help** and made **zero searches**.

## Run locally

- Use Python 3.12 and open PowerShell in the project folder.
- Create the environment with `py -3.12 -m venv .venv`.
- Install dependencies with `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.
- If `.env` does not exist, create it with `Copy-Item .env.example .env`.
- Start Ollama and run `ollama pull qwen3:4b` if the model is missing.
- Start the app with `.\.venv\Scripts\python.exe -m streamlit run app.py`.
- Open [the local app](http://127.0.0.1:8501). Select **Live AI** for the assignment demo and use **Semantic** search on a text-based research PDF.
- For an API provider, add the matching variable to `.env` and select the provider in the sidebar. Use `OPENAI_API_KEY`, `GOOGLE_API_KEY`, or `GROQ_API_KEY`. Git ignores `.env`.
- For a quick walkthrough, choose **Evidence preview** and **Try a synthetic sample**. This mode displays excerpts without calling a model.

## Workspace

- **Paper chat** answers questions and displays evidence, page citations, and prompt traces.
- **Build a deck** creates 4–10 content slides and an editable PowerPoint with notes and source pages.
- **Viva rehearsal** asks an examiner question and gives advisory feedback on accuracy, depth, and clarity in Live AI mode.
- **Session record** displays activity counts and exports a Markdown report.

## Credits

- Workspace inspiration from Rahul J’s [Paperloom Research Studio](https://github.com/SolvyrEryx/paperloom-research-studio).
