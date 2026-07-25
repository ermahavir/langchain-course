from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

from graph.chains.retrieval_grader import retrieval_grader, GradeDocuments
from ingestion import retriever
from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import hallucination_chain, GradeHallucination
from graph.chains.router import question_router, RouteQuery

def test_retrieval_grader_yes() -> None:
    question = "ai agent"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content
    res = retrieval_grader.invoke({"question": question, "document": doc_txt})
    assert res.binary_score == "Yes"


def test_retrieval_grader_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content
    res = retrieval_grader.invoke({"question": "How to make pizza", "document": doc_txt})
    assert res.binary_score == "No"


def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    res = generation_chain.invoke({"question": question, "context": docs})
    pprint(res)

def test_hallucination_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content

    generation = generation_chain.invoke({"question": question, "context": doc_txt})
    res: GradeHallucination = hallucination_chain.invoke({"documents": doc_txt, "generation": generation})
    assert res.binary_score

def test_hallucination_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content

    res: GradeHallucination = hallucination_chain.invoke({
        "documents": doc_txt, 
        "generation": "Pizza is a type of food that is made from dough and cheese."
        })
    assert not res.binary_score


def test_router_to_vectorstore() -> None:
    question = "agent memory"
    res : RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "vectorstore"

def test_router_to_websearch() -> None:
    question = "how to make pizza"
    res : RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "websearch"