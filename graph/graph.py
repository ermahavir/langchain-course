# The orchestrator graph

from dotenv import load_dotenv

from langgraph.graph import StateGraph, END

from graph.consts import RETRIEVE, GRADE_DOCUMENTS, WEBSEARCH, GENERATE
from graph.nodes import retrieve, grade_documents, web_search, generate
from graph.state import GraphState
from graph.chains.answer_grader import answer_grader_chain
from graph.chains.hallucination_grader import hallucination_chain
from graph.chains.router import question_router, RouteQuery


load_dotenv()

def route_question(state: GraphState) -> str:
    print("--- ROUTING QUESTION ---")
    question = state["question"]
    res : RouteQuery = question_router.invoke({"question": question})
    if res.datasource == "vectorstore":
        print("--- DECISION: ROUTING TO VECTORSTORE ---")
        return RETRIEVE
    else:
        print("--- DECISION: ROUTING TO WEBSEARCH ---")
        return WEBSEARCH

def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("--- CHECK HAALLUCINATION ---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_chain.invoke({"documents": documents, "generation": generation})
    if hallucination_grade := score.binary_score:
        print("--- GENERATION IS GROUNDED IN DOCUMENTS ---")
        print("--- GRADE GENERATION VS QUESTION ---")
        score = answer_grader_chain.invoke({"question": question, "generation": generation})
        if answer_grade := score.binary_score:
            print("--- DECISION: ANSWER IS USEFUL ---")
            return "useful"
        else:
            print("--- DECISION: ANSWER DOES NOT ANSWERS THE QUESTION ---")
            return "not_useful"
    else:
        print("--- DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS ---")
        return "not_supported"

def decide_to_generate(state: GraphState) -> str:
    print("--- ASSESS GRADED DOCUMENTS ---")
    if state["web_search"]:
        print("--- DECISION: NOT ALL DOCUMENTS ARE RELEVANT ---")
        return WEBSEARCH
    else:
        print("--- DECISION: GENERATE ---")
        return GENERATE

workflow = StateGraph(GraphState)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(WEBSEARCH, web_search)
workflow.add_node(GENERATE, generate)


workflow.set_conditional_entry_point(route_question, {RETRIEVE: RETRIEVE, WEBSEARCH: WEBSEARCH})

#workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(GRADE_DOCUMENTS, decide_to_generate, {WEBSEARCH: WEBSEARCH, GENERATE: GENERATE})
workflow.add_conditional_edges(GENERATE, grade_generation_grounded_in_documents_and_question,
 {"not_supported": GENERATE, "not_useful": WEBSEARCH, "useful": END})
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")

