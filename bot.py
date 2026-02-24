import os
import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def extract_exact_answer(question: str, raw_sources: list) -> str:
    """Pure extraction from PDF — returns complete text."""
    stopwords = {"what","is","the","a","an","are","how","does","do","in","of",
                "for","and","to","this","that","which","tell","me","about",
                "explain","describe","give","define","list","please","can",
                "you","was","were","has","have","had","will","would","could",
                "should","its","it","be","been","or","but","if","then","there"}

    keywords = [w for w in question.lower().split()
               if w not in stopwords and len(w) > 2]

    # Combine ALL chunk content fully
    all_text = "\n".join([doc.page_content.strip() for doc in raw_sources])

    if not keywords:
        return all_text

    # Score and return best matching sentences
    all_sentences = []
    for line in all_text.split('\n'):
        for s in line.split('.'):
            s = s.strip()
            if len(s) > 15:
                all_sentences.append(s)

    def score(s):
        return sum(1 for k in keywords if k in s.lower())

    top = sorted(all_sentences, key=score, reverse=True)
    top = [s for s in top if score(s) > 0]

    if not top:
        return all_text

    return ". ".join(top) + "."


class GroqLLM(LLM):
    model: str = "llama-3.1-8b-instant"

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self):
        return "groq"

    @property
    def _identifying_params(self):
        return {"model": self.model}

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
        if not GROQ_API_KEY:
            return "__EXTRACT__"
        try:
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            # Extract question and context from prompt
            context = ""
            question = ""
            if "Context:" in prompt and "Question:" in prompt:
                context = prompt.split("Context:")[1].split("Question:")[0].strip()
                question = prompt.split("Question:")[1].split("Answer:")[0].strip()
            else:
                question = prompt

            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a document assistant. Answer questions using ONLY the provided context. Give complete, detailed answers including ALL points mentioned. Do not add outside information."
                    },
                    {
                        "role": "user",
                        "content": f"Context from document:\n{context}\n\nQuestion: {question}\n\nGive a complete answer with ALL points from the document:"
                    }
                ],
                "max_tokens": 1024,
                "temperature": 0.1
            }
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            if r.status_code == 200:
                result = r.json()
                text = result["choices"][0]["message"]["content"].strip()
                if text and len(text) > 10:
                    return text
            return "__EXTRACT__"
        except Exception:
            return "__EXTRACT__"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def load_faiss(path):
    return FAISS.load_local(path, get_embeddings(), allow_dangerous_deserialization=True)


PROMPT = PromptTemplate(
    template="""Answer the question using ONLY the text from the context below.
Give a COMPLETE answer. Include ALL bullet points and details mentioned.
Do NOT skip anything. Do NOT add outside information.
If not found say: "This information is not in the document."

Context:
{context}

Question: {question}

Complete Answer (include ALL points):""",
    input_variables=["context", "question"]
)


def load_chain_approach1():
    vs = load_faiss("storage/approach1/faiss_index")
    retriever = vs.as_retriever(search_kwargs={"k": 32})
    return RetrievalQA.from_chain_type(
        llm=GroqLLM(), chain_type="stuff", retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}, return_source_documents=True
    )


def load_chain_approach2():
    vs = load_faiss("storage/approach2/faiss_index")
    retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 22, "fetch_k": 22})
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history", return_messages=True, output_key="answer", k=3
    )
    return ConversationalRetrievalChain.from_llm(
        llm=GroqLLM(), retriever=retriever, memory=memory,
        return_source_documents=True, verbose=False
    )


def load_chain_approach3():
    vs = load_faiss("storage/approach3/faiss_index")
    retriever = vs.as_retriever(search_kwargs={"k": 17})
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history", return_messages=True, output_key="answer", k=5
    )
    return ConversationalRetrievalChain.from_llm(
        llm=GroqLLM(), retriever=retriever, memory=memory,
        return_source_documents=True, verbose=False
    )


build_approach1_chain = load_chain_approach1
build_approach2_chain = load_chain_approach2
build_approach3_chain = load_chain_approach3


def query_chain(chain, question: str, approach_num: int) -> dict:
    try:
        chain_type = type(chain).__name__
        if "RetrievalQA" in chain_type:
            result = chain({"query": question})
            answer = result.get("result", "")
            raw_sources = result.get("source_documents", [])
        else:
            result = chain({"question": question})
            answer = result.get("answer", "")
            raw_sources = result.get("source_documents", [])

        if (not answer or len(answer.strip()) < 15
                or "__EXTRACT__" in answer
                or "Check your" in answer):
            answer = extract_exact_answer(question, raw_sources)

        sources = []
        for doc in raw_sources[:5]:
            if hasattr(doc, "page_content"):
                sources.append({
                    "text": doc.page_content.strip(),
                    "page": doc.metadata.get("page", "?") if hasattr(doc, "metadata") else "?"
                })

        return {"answer": answer, "sources": sources}

    except Exception as e:
        err = str(e)
        try:
            if "query" in err:
                result = chain({"query": question})
                raw = result.get("source_documents", [])
                return {"answer": extract_exact_answer(question, raw), "sources": []}
            if "question" in err:
                result = chain({"question": question})
                raw = result.get("source_documents", [])
                return {"answer": extract_exact_answer(question, raw), "sources": []}
        except Exception:
            pass
        return {"answer": f"Error: {err}", "sources": []}