from typing import Any, Dict

from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState

def grade_documents(state: GraphState) -> Dict[str, Any]:
    print("--- GRADE DOCUMENTS NODE ---")
    question = state["question"]
    documents = state["documents"]
    
    filtered_docs = []
    web_search = False

    for doc in documents:
        score = retrieval_grader.invoke({"question": question, "document": doc.page_content})   
        grade = score.binary_score
        if grade.lower() == "yes":
            filtered_docs.append(doc)
        else:
            web_search = True
            continue

    return {"documents": filtered_docs, "web_search": web_search}