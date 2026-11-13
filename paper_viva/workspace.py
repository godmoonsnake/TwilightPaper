import re

from .agents import AgentResult, Intent, SearchTrace, Slide, SlideDeck
from .paper import PaperIndex, Passage, index_passages

SAMPLE = (
    "Research question and motivation. Can a retrieval index help students find evidence in lab "
    "manuals faster? The fictional Lab Atlas study compares page search with reading a contents "
    "list. All participants and results in this example are invented for a classroom demo.",
    "Method and experimental design. In this fictional study, 24 students used both interfaces "
    "in alternating order. They answered six questions per interface and recorded the page that "
    "supported each answer. The experiment recorded completion time and citation accuracy.",
    "Results and limitations. In this invented example, median time was 71 seconds with search "
    "and 103 seconds with the contents list. Citation accuracy was similar. The study used one "
    "manual and did not measure long-term learning, so the finding may not generalize.",
    "Interpretation and future work. The simulated results suggest that quicker evidence "
    "navigation can save time. A follow-up could compare results across different subjects and "
    "test whether students still recall the material one week later.",
)


def load_sample() -> PaperIndex:
    passages = [Passage(f"p{i}-1", i, text) for i, text in enumerate(SAMPLE, 1)]
    return index_passages("Lab Atlas (synthetic sample)", len(passages), passages, lexical=True)


def preview(request: str, paper: PaperIndex | None, intent: Intent, count: int | None) -> AgentResult:
    if paper is None:
        return AgentResult(intent, "Index a PDF or load the sample to preview evidence.", [])
    query = request.split("Focus on:", 1)[-1].strip() if intent == "slides" else request
    question = ""
    if intent == "viva":
        if request.startswith("Compare my answer"):
            query = request.split("\n\n", 1)[-1]
            question = "Compare your answer with the excerpts below."
        elif "critical evaluation" in request:
            query = "limitations assumptions generalizability future work"
            question = "Which limitation matters most, and how would you address it?"
        elif "methods and results" in request:
            query = "method experimental design measured results"
            question = "What did the authors measure, and how does it support their conclusion?"
        else:
            query = "research question motivation contribution"
            question = "What problem does this study address, and why does it matter?"
    hits = tuple(paper.search(query, limit=count or 4))
    trace = [SearchTrace(query, hits)]
    label = "Extractive preview · no model was called"
    if intent == "slides":
        slides = [
            Slide(title=f"Evidence {i}", bullets=[excerpt(p.text)], pages=[p.page], notes=p.text[:450])
            for i, p in enumerate(hits, 1)
        ]
        deck = SlideDeck(title="Evidence preview", slides=slides)
        return AgentResult(intent, label, [], trace, deck)
    heading = question or "Relevant excerpts from the selected paper."
    evidence = "\n\n".join(f"{p.text} [p. {p.page}]" for p in hits)
    return AgentResult(intent, f"**{label}**\n\n{heading}\n\n{evidence}", [], trace)


def excerpt(text: str) -> str:
    sentence = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]
    if len(sentence) <= 150:
        return sentence
    return sentence[:147].rsplit(" ", 1)[0] + "…"


def session_report(filename: str, entries: list[tuple[str, AgentResult, str]]) -> str:
    searches = sum(len(result.searches) for _, result, _ in entries)
    lines = ["# Research session", "", f"Paper: {filename}", "",
             f"Completed tasks: {len(entries)} · ChromaDB searches: {searches}"]
    for i, (request, result, mode) in enumerate(entries, 1):
        pages = sorted({p.page for search in result.searches for p in search.passages})
        lines.extend(["", f"## {i}. {result.intent.capitalize()} · {mode}", "", request, "",
                      result.text, "", f"Retrieved PDF pages: {', '.join(map(str, pages)) or 'none'}"])
        if result.deck:
            for slide in result.deck.slides:
                lines.extend(["", f"### {slide.title}", *[f"- {bullet}" for bullet in slide.bullets],
                              "", slide.notes, f"Pages: {', '.join(map(str, slide.pages))}"])
    return "\n".join(lines) + "\n"
