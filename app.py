from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import streamlit as st

from paper_viva.agents import AGENT_NAMES, AgentError, AgentResult, Intent, check_connection, run_request
from paper_viva.models import PROVIDER_LABELS, ModelConfigurationError, ModelSettings
from paper_viva.paper import EMBEDDING_MODEL, PaperError, PaperIndex, index_paper
from paper_viva.slides import build_powerpoint
from paper_viva.workspace import load_sample, preview, session_report

EXAMPLE_REQUESTS: dict[str, tuple[Intent, str]] = {
    "Summarize": ("summary", "Summarize the paper's motivation, methods, main findings, and limitations."),
    "Explain a concept": ("concepts", "Explain the paper's main method in simple terms."),
    "Make slides": ("slides", "Create a PowerPoint covering the problem, method, and findings."),
    "Practice viva": ("viva", "Give me five likely viva questions and suggested answers."),
}
PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
WORKSPACE_CSS = Path(__file__).with_name("workspace.css")
ACTION_ICONS = {
    "summary": ":material/summarize:",
    "concepts": ":material/lightbulb:",
    "slides": ":material/slideshow:",
    "viva": ":material/forum:",
}


@dataclass
class Turn:
    request: str
    result: AgentResult
    powerpoint: bytes | None = None
    workspace: str = "read"
    mode: str = "Live AI"


def model_error(error: Exception) -> str:
    status = getattr(error, "status_code", None) or getattr(error, "code", None)
    if status in {401, 403}:
        return "The provider rejected the credentials. Check the API key and model access."
    if status == 404:
        return "The provider could not find this model. Check the model name and access for your API key."
    if status == 429:
        return "The model provider reported a rate or quota limit. Wait or choose another model."
    return (
        f"The model or retrieval service did not complete the request ({type(error).__name__}). "
        "Check your connection and try again."
    )


def replace_paper(paper: PaperIndex | None, identity=None) -> None:
    previous = st.session_state.get("paper")
    if previous is not None:
        previous.delete()
    st.session_state.paper = paper
    st.session_state.paper_identity = identity
    st.session_state.turns = []


def sidebar() -> tuple[ModelSettings | None, PaperIndex | None, str]:
    settings = None
    with st.sidebar:
        st.subheader("Research workspace")
        st.caption("Set up your paper and model.")
        mode = st.segmented_control(
            "Response mode", ["Live AI", "Evidence preview"], default="Live AI",
            selection_mode="single", required=True,
            help="Preview shows excerpts without using a language model.",
        )
        try:
            default = ModelSettings.from_env()
            providers = list(PROVIDER_LABELS)
            provider = st.selectbox("Model provider", providers, index=providers.index(default.provider),
                                    format_func=PROVIDER_LABELS.get)
            settings = ModelSettings.from_env(provider)
            st.write(settings.name)
            model_check_key = f"model_check:{settings.provider}:{settings.name}"
            if st.button("Test model connection", width="stretch"):
                st.session_state[model_check_key] = None
                try:
                    with st.spinner("Calling the model…"):
                        answer = check_connection(settings.connect())
                    st.session_state[model_check_key] = (
                        bool(answer),
                        answer or "The model returned an empty response.",
                    )
                except ModelConfigurationError as exc:
                    st.session_state[model_check_key] = (False, str(exc))
                except Exception as exc:
                    st.session_state[model_check_key] = (False, model_error(exc))
            if st.session_state.get(model_check_key):
                connected, detail = st.session_state[model_check_key]
                if connected:
                    st.success("Connected · model responded successfully.")
                else:
                    st.error(detail)
        except ModelConfigurationError as exc:
            st.error(str(exc))

        st.divider()
        st.header("Research paper", divider="blue")
        upload = st.file_uploader("Upload a PDF with selectable text", type="pdf")
        with st.expander("PDF search settings"):
            indexing = st.radio("PDF search", ["Semantic", "Keyword"], horizontal=True)
        if st.button("Try a synthetic sample", icon=":material/description:", width="stretch"):
            replace_paper(load_sample())
            st.rerun()
        paper = st.session_state.get("paper")
        if upload is not None:
            pdf_bytes = upload.getvalue()
            identity = (upload.name, sha256(pdf_bytes).hexdigest(), indexing)
            if identity != st.session_state.get("paper_identity"):
                if st.button("Index paper", type="primary", width="stretch"):
                    try:
                        with st.spinner("Reading the PDF and creating embeddings…"):
                            replacement = index_paper(upload.name, pdf_bytes, lexical=indexing == "Keyword")
                        replace_paper(replacement, identity)
                        st.rerun()
                    except PaperError as exc:
                        st.error(str(exc))
                    except Exception as exc:
                        st.error(
                            f"Indexing failed ({type(exc).__name__}). The first run needs "
                            "internet access to download ChromaDB's embedding model."
                        )
                st.caption("The uploaded file becomes active after indexing.")
        if paper is not None:
            st.success(f"Active paper · {paper.filename}")
            st.caption(
                f"{paper.page_count} PDF pages · {paper.passage_count} passages in ChromaDB"
            )
            st.caption("Keyword vectors" if paper.lexical else f"Embeddings · {EMBEDDING_MODEL}")
            if st.button("Clear paper and conversation", width="stretch"):
                replace_paper(None)
                st.rerun()
        if settings and settings.provider != "ollama":
            st.caption("Requests and retrieved excerpts are sent to the selected model provider.")
    return settings, paper, mode


def render_trace(result: AgentResult) -> None:
    with st.expander("Run details · prompts and retrieval"):
        source = "Detected" if any(stage.stage == "Intent router" for stage in result.prompts) else "Selected"
        st.write(f"**{source} intent:** `{result.intent}`  ·  **Handler:** {AGENT_NAMES[result.intent]}")
        st.write(f"**ChromaDB searches:** {len(result.searches)}")
        for search in result.searches:
            st.caption("Fallback similarity search query" if search.fallback else "Similarity search query")
            st.text(search.query)
            for passage in search.passages:
                st.markdown(f"**PDF page {passage.page} · passage {passage.id}**")
                st.text(passage.text)
        for stage in result.prompts:
            st.markdown(f"#### {stage.stage}")
            for index, message in enumerate(stage.messages):
                if message.type == "system":
                    label = "System prompt"
                elif message.type == "human":
                    is_example = stage.stage == "Intent router" and index < len(stage.messages) - 1
                    label = "Few-shot example · user" if is_example else "User prompt"
                elif message.type == "ai" and stage.stage == "Intent router":
                    label = "Few-shot example · assistant"
                else:
                    continue
                st.caption(label)
                st.code(message.content, language=None)


def render_turn(turn: Turn, index: int, paper: PaperIndex | None) -> None:
    with st.chat_message("user"):
        st.write(turn.request)
    with st.chat_message("assistant"):
        result = turn.result
        st.caption(AGENT_NAMES[result.intent])
        if turn.mode != "Live AI":
            st.caption("Evidence preview · no model was called")
        if result.deck and turn.powerpoint and paper:
            st.subheader(result.deck.title)
            for slide in result.deck.slides:
                st.markdown(f"**{slide.title}**")
                for bullet in slide.bullets:
                    st.write(f"• {bullet}")
                st.caption("PDF pages " + ", ".join(map(str, sorted(set(slide.pages)))))
                if slide.notes:
                    st.caption(f"Speaker notes: {slide.notes}")
            st.download_button(
                "Download PowerPoint",
                turn.powerpoint,
                file_name=f"{Path(paper.filename).stem}-presentation.pptx",
                mime=PPTX_MIME,
                key=f"download-{index}",
                on_click="ignore",
            )
        else:
            st.markdown(result.text)
        render_trace(result)


def main() -> None:
    st.set_page_config(page_title="TwilightPaper", page_icon=":material/menu_book:", layout="wide")
    st.html(WORKSPACE_CSS)
    if "turns" not in st.session_state:
        st.session_state.turns = []

    settings, paper, mode = sidebar()
    with st.container(key="workspace_hero"):
        st.title("TwilightPaper")
        st.write("Understand a research paper, build your slide deck, and practise for your viva.")
    if paper is None:
        st.info("Upload and index a PDF from the sidebar, or try the sample paper.", icon=":material/upload_file:")
    else:
        st.caption(f"Active paper · {paper.filename} · {paper.page_count} pages")

    if mode == "Evidence preview":
        st.info("Evidence preview shows excerpts and sample activities. Select Live AI to demonstrate the model.")
    request = None
    selected_intent = None
    slide_count = None
    workspace = "read"
    error_slot = st.empty()
    read_tab, slides_tab, viva_tab, report_tab = st.tabs(
        ["Paper chat", "Build a deck", "Viva rehearsal", "Session record"]
    )
    turns = st.session_state.turns

    with read_tab:
        with st.container(key="paper_actions"):
            columns = st.columns(len(EXAMPLE_REQUESTS))
            for column, (label, (intent, example)) in zip(columns, EXAMPLE_REQUESTS.items()):
                if column.button(label, icon=ACTION_ICONS[intent], width="stretch", disabled=paper is None):
                    request, selected_intent = example, intent
        for index, turn in enumerate(turns):
            if turn.workspace == "read":
                render_turn(turn, index, paper)
        typed = st.chat_input("Ask about your paper or request slides or viva questions", max_chars=1500)
        if typed and typed.strip():
            request, selected_intent = typed.strip(), None

    with slides_tab:
        st.subheader("Build your presentation")
        with st.form("deck_options"):
            focus = st.text_input("Presentation focus", placeholder="Method, results, and limitations", max_chars=300)
            count = st.slider("Content slides", 4, 10, 5, help="The title slide is added separately.")
            if st.form_submit_button("Prepare slides", type="primary", disabled=paper is None):
                topic = focus.strip() or "the problem, methods, findings, and reported limitations"
                request = f"Create exactly {count} content slides, plus a title slide. Focus on: {topic}."
                selected_intent, workspace, slide_count = "slides", "slides", count
        latest = next(((i, t) for i, t in reversed(list(enumerate(turns))) if t.workspace == "slides"), None)
        if latest:
            render_turn(latest[1], latest[0], paper)

    with viva_tab:
        st.subheader("Rehearse for your viva")
        depth = st.selectbox("Question depth", ["Foundations", "Methods and results", "Critical evaluation"])
        if st.button("Ask an examiner question", type="primary", disabled=paper is None):
            request = (
                f"Ask exactly one viva question on {depth.lower()}. Search the paper for the topic. "
                "Append one supporting page citation like [p. 3]. Do not give the answer yet."
            )
            selected_intent, workspace = "viva", "question"
        latest_question = next((i for i in reversed(range(len(turns))) if turns[i].workspace == "question"), None)
        if latest_question is not None:
            question = turns[latest_question]
            render_turn(question, latest_question, paper)
            with st.form(f"viva_answer_{latest_question}"):
                answer = st.text_area("Your answer", max_chars=1500)
                can_score = mode == "Live AI" and question.mode == "Live AI"
                if st.form_submit_button("Get advisory feedback" if can_score else "Compare with evidence"):
                    if answer.strip():
                        request = (
                            "Evaluate this answer against the paper.\n\n"
                            f"Examiner question: {question.result.text[:1200]}\n\n"
                            f"Candidate answer: {answer.strip()}"
                        )
                        selected_intent, workspace = "viva", "feedback"
                        if not can_score:
                            request = f"Compare my answer with the source.\n\n{answer.strip()}"
                            workspace = "self-review"
                    else:
                        st.info("Write your answer before submitting.")
            for i, turn in enumerate(turns[latest_question + 1:], latest_question + 1):
                if turn.workspace in {"feedback", "self-review"}:
                    render_turn(turn, i, paper)

    with report_tab:
        tasks, searches, attempts = st.columns(3)
        tasks.metric("Completed tasks", len(turns), border=True)
        searches.metric("Paper searches", sum(len(t.result.searches) for t in turns), border=True)
        attempts.metric("Viva attempts", sum(t.workspace in {"feedback", "self-review"} for t in turns), border=True)
        if turns:
            report = session_report(paper.filename if paper else "No paper", [
                (turn.request, turn.result, turn.mode) for turn in turns
            ])
            st.download_button("Download session report", report, "research-session.md", "text/markdown")
            for turn in reversed(turns):
                st.caption(f"{AGENT_NAMES[turn.result.intent]} · {turn.mode}")
                st.text(turn.request[:250])

    if request and (settings is not None or mode == "Evidence preview"):
        history = "\n\n".join(
            f"User: {turn.request[:1500]}\nHandler: {AGENT_NAMES[turn.result.intent]}"
            f"\nAssistant: {turn.result.text[:850]}"
            for turn in st.session_state.turns[-3:]
        )
        try:
            run_mode = "Evidence preview" if workspace == "self-review" else mode
            with error_slot, st.spinner("Preparing the response…"):
                if run_mode == "Live AI":
                    result = run_request(settings.connect(), request, paper, history,
                                         selected_intent=selected_intent, slide_count=slide_count)
                else:
                    result = preview(request, paper, selected_intent or "concepts", slide_count)
                powerpoint = build_powerpoint(result.deck, paper.filename) if result.deck and paper else None
            turns.append(Turn(request, result, powerpoint, workspace, run_mode))
            st.rerun()
        except (ModelConfigurationError, AgentError) as exc:
            error_slot.error(str(exc))
        except Exception as exc:
            error_slot.error(model_error(exc))


if __name__ == "__main__":
    main()
