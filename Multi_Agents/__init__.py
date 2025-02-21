from pathlib import Path
import sys
import os

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from typing import TypedDict

from APIs import APIManager
from Multi_Agents.agents import Analyzer_Agent, Judge_Agent
from langgraph.graph import StateGraph, END, START

import graphviz


class CV_Analyzer:
    # Define a state object like the session for all the graph
    class AgentState(TypedDict):
        input: str
        result1: str
        result2: str
        result3: str
        final_response: str

    def __init__(self):
        # Initialize and keep warm the apis
        manager = APIManager(['llama', 'phi', 'mistral', 'qwen'])
        manager.start_and_keep_warm()

        # Define the agents
        self.analyzer1_agent = Analyzer_Agent('llama-app', 'Llama') # Analyzer Agent
        self.analyzer2_agent = Analyzer_Agent('qwen-app', 'Qwen') # Analyzer Agent
        self.analyzer3_agent = Analyzer_Agent('phi-app', 'Phi') # Analyzer Agent
        self.judge_agent = Judge_Agent('mistral-app', 'Mistral') # Judge Agent

        self.workflow = StateGraph(self.AgentState)
        self._build()

        # Get graph and visualize
        graph_data = self.workflow.get_graph()
        self.save_graph(graph_data)

    def _build(self):
        # Add analyzer nodes
        self.workflow.add_node("analyzer_1", lambda state: {"result1": self.analyzer1_agent.invoke(state["input"])})
        self.workflow.add_node("analyzer_2", lambda state: {"result2": self.analyzer2_agent.invoke(state["input"])})
        self.workflow.add_node("analyzer_3", lambda state: {"result3": self.analyzer3_agent.invoke(state["input"])})

        # Trigger all analyzers from the starting node
        self.workflow.add_edge(START, "analyzer_1")
        self.workflow.add_edge(START, "analyzer_2")
        self.workflow.add_edge(START, "analyzer_3")

        # Add node to check if all results are ready
        self.workflow.add_node("check_results", lambda state: state)

        # Add judge node
        self.workflow.add_node("judge", lambda state: {"final_response": self.judge_agent.invoke([state["result1"], state["result2"], state["result3"]])})

        # Set up parallel execution flow
        self.workflow.add_edge("analyzer_1", "check_results")
        self.workflow.add_edge("analyzer_2", "check_results")
        self.workflow.add_edge("analyzer_3", "check_results")

        self.workflow.add_conditional_edges(
            "check_results",
            lambda state: "judge" if all([
                state.get("result1"),
                state.get("result2"),
                state.get("result3")
            ]) else "check_results",
            { "judge": "judge", "check_results": "check_results"}
        )

        self.workflow.add_edge("judge", END)
        self.workflow = self.workflow.compile()

    def save_graph(self, graph_data):
        """Saves the graph structure as an image and removes extra files."""
        dot = graphviz.Digraph(format='png')  # Set output format
        
        # Add nodes
        for node_id, node in graph_data.nodes.items():
            dot.node(node_id, node_id)  # Label nodes with their names
        
        # Add edges
        for edge in graph_data.edges:
            dot.edge(edge.source, edge.target)  # Draw connections

        # Save the graph as an image only (avoid the .dot file)
        output_path = "Multi_Agents/graphs/cv_analyzer"
        dot.render(output_path, format="png", cleanup=True)  # Saves 'graph_diagram.png' & deletes '.dot'

    def analyze(self, cv: str):
        initial_state = self.AgentState(
            input=cv,
            result1=None,
            result2=None,
            result3=None,
            final_response=None
        )
        # Run the workflow
        final_state = self.workflow.invoke(initial_state)
        return final_state["final_response"]
