import google.generativeai as genai
from langchain_core.prompts import PromptTemplate
from retriever import retrieve_chunks
from dotenv import load_dotenv
import os

load_dotenv()

llm = genai.GenerativeModel("gemini-2.5-flash")

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

RAG_PROMPT = """
You are a helpful assistant that answers questions based strictly on the supplied context.

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
- If you cannot answer from the context, say "I cannot answer from the document."
- Keep answers concise.
- Do NOT include information outside the context.
- Cite page numbers when possible.

ANSWER:
"""

prompt = PromptTemplate(
    template=RAG_PROMPT,
    input_variables=['context', 'question']
)


def answer_question(query, store=None, k=10):
    """
    Retrieve chunks, generate answer using Gemini, add citations.

    Args:
        query (str): The user's question.
        store (FAISS): The FAISS vector store.
        k (int): Number of top chunks to retrieve.

    Returns:
        tuple: (answer string, list of citations)
    """

    # Retrieve chunks
    context, metas = retrieve_chunks(query, store=store, k=k)

    # 2. Format context
    combined = "\n\n".join(context)

    formatted_prompt = prompt.format(
        context=combined,
        question=query
    )

    response = llm.generate_content(formatted_prompt)

    answer = response.text

    # 5. Citations
    citations = []
    for m in metas:
        citations.append(f"Page {m.get('page')} [{m.get('type')}]")

    print("\n---- CONTEXT SENT TO GEMINI ----\n")
    print(combined)
    print("\n-------------------------------\n")


    return answer, citations

