"""Answer generation with Gemini through LangChain. Answers cite chunks as [n]."""
from collections.abc import Iterator

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions using ONLY the numbered context passages. "
            "Cite the passages you use like [1] or [2][3]. "
            "If the context does not contain the answer, say you could not find it "
            "in the documents. Do not use outside knowledge.",
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


def format_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {c['source']})\n{c['text']}" for i, c in enumerate(chunks, 1)
    )


class Generator:
    def __init__(self) -> None:
        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0,
        )
        self.chain = PROMPT | llm | StrOutputParser()

    def answer(self, question: str, chunks: list[dict]) -> str:
        return self.chain.invoke(
            {"context": format_context(chunks), "question": question}
        )

    def stream(self, question: str, chunks: list[dict]) -> Iterator[str]:
        yield from self.chain.stream(
            {"context": format_context(chunks), "question": question}
        )
