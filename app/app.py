"""
Gradio Demo — RAG PDF Assistant
--------------------------------
Upload any PDF, ask questions, get answers grounded in the document.
"""

import os
import traceback

import gradio as gr

from rag_pipeline import (
    build_rag_chain,
    build_vectorstore,
    load_and_split_pdf,
    load_llm,
)

# ----- Global state (loaded once per session) -------------------------------
_vectorstore = None
_chain = None
_llm = None
_doc_name = None


def load_pdf(pdf_file):
    """Process uploaded PDF and build the RAG chain."""
    global _vectorstore, _chain, _llm, _doc_name

    if pdf_file is None:
        return " Please upload a PDF first.", gr.update(interactive=False)

    try:
        _doc_name = os.path.basename(pdf_file)
        chunks = load_and_split_pdf(pdf_file)

        _vectorstore = build_vectorstore(chunks)

        if _llm is None:
            _llm = load_llm()

        _chain = build_rag_chain(_vectorstore, _llm)

        return (
            f" **{_doc_name}** loaded — {len(chunks)} chunks indexed. Ask away!",
            gr.update(interactive=True),
        )
    except Exception as e:
        traceback.print_exc()
        return f" Error: {type(e).__name__}: {str(e)}", gr.update(interactive=False)


def answer_question(question, history):
    """Run a query through the RAG chain and return answer + sources."""
    global _chain

    if history is None:
        history = []

    if not question or not question.strip():
        return history, ""

    if _chain is None:
        history.append({"role": "user", "content": question})
        history.append(
            {"role": "assistant", "content": " Please upload and process a PDF first."}
        )
        return history, ""

    try:
        result = _chain.invoke({"query": question})
        answer = str(result["result"]).strip()
        sources = result["source_documents"]

        source_text = "\n\n---\n**Sources:**\n"
        for doc in sources[:3]:
            page = doc.metadata.get("page", "?")
            snippet = doc.page_content[:200].replace("\n", " ")
            source_text += f"\n *Page {page}*: {snippet}..."

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer + source_text})
        return history, ""

    except Exception as e:
        traceback.print_exc()
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": f" Error: {str(e)}"})
        return history, ""


# --- UI ----------------------------------------------
_THEME = gr.themes.Soft(primary_hue="blue")
_CSS = """
.status-box { border-radius: 8px; padding: 12px; }
.gr-button-primary { background: #2563eb !important; }
footer { display: none !important; }
"""

with gr.Blocks(title="RAG PDF Assistant", theme=_THEME, css=_CSS) as demo:

    gr.Markdown(
        """
        # RAG PDF Assistant
        **Upload a PDF → Ask questions → Get answers grounded in your document.**

        *Powered by FAISS + sentence-transformers + Qwen2.5 (fully local, no API keys)*
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Upload your PDF")
            pdf_input = gr.File(
                label="PDF File",
                file_types=[".pdf"],
                type="filepath",
            )
            load_btn = gr.Button(" Process PDF", variant="primary")
            status = gr.Markdown("*No document loaded yet.*")

        with gr.Column(scale=2):
            gr.Markdown("### 2. Ask questions")
            chatbot = gr.Chatbot(
                label="Conversation",
                height=420,
                value=[],
                type="messages",  # use dict messages, not tuples
            )
            with gr.Row():
                question_box = gr.Textbox(
                    placeholder="e.g. What is the main topic of this document?",
                    label="Your question",
                    scale=4,
                    interactive=False,  # gated until a PDF is loaded
                )
                submit_btn = gr.Button("Ask →", variant="primary", scale=1)

            clear_btn = gr.Button(" Clear chat", size="sm")

    gr.Markdown("### Example questions to try")
    gr.Examples(
        examples=[
            ["What is the main topic of this document?"],
            ["Summarize the key findings."],
            ["What conclusions does the author draw?"],
            ["List the main sections covered."],
        ],
        inputs=question_box,
    )

    # ------- Events -------------------------
    load_btn.click(
        fn=load_pdf,
        inputs=[pdf_input],
        outputs=[status, question_box],
    )

    submit_btn.click(
        fn=answer_question,
        inputs=[question_box, chatbot],
        outputs=[chatbot, question_box],
    )

    question_box.submit(
        fn=answer_question,
        inputs=[question_box, chatbot],
        outputs=[chatbot, question_box],
    )

    clear_btn.click(lambda: [], outputs=[chatbot])


if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
