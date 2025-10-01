from typing import Annotated, Tuple, List, Optional
from typing_extensions import TypedDict
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from tools import advanced_image_processor, object_detection_tool, image_analysis_tool, image_enhancement_tool
from langchain.chat_models import init_chat_model
import logging

# Set up logging
logger = logging.getLogger(__name__)

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    input_data: list[dict]
    current_variables: dict
    intermediate_outputs: list[dict]
    output_image_paths: list[str]
    analysis_results: dict
    image_path: str

def make_tool_graph():
    llm = init_chat_model("groq:llama3-8b-8192")
    tools = [advanced_image_processor, object_detection_tool, image_analysis_tool, image_enhancement_tool]
    tool_node = ToolNode(tools)
    llm_with_tools = llm.bind_tools(tools)

    def call_llm_model(state: State):
        system_prompt = """
        You are an expert computer vision AI assistant specializing in image processing and analysis.
        You have access to tools for:
        1. advanced_image_processor: Execute custom OpenCV/PIL code
        2. object_detection_tool: Detect objects, faces, edges, and contours
        3. image_analysis_tool: Analyze image properties, statistics, histograms
        4. image_enhancement_tool: Apply enhancement techniques

        Common operations:
        - Grayscale: `gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY); output_images.append(gray)`
        - Blur: `blurred = cv2.GaussianBlur(image, (15, 15), 0); output_images.append(blurred)`
        - Edge Detection: `edges = cv2.Canny(gray, 100, 200); output_images.append(edges)`
        - Resize: `resized = cv2.resize(image, (300, 300)); output_images.append(resized)`

        Always:
        1. Use the image_path from state['image_path']
        2. Explain your approach in the `thought` parameter
        3. Append results to `output_images` list
        4. Handle errors gracefully
        5. Provide meaningful analysis when requested

        For a request like "Apply blur", use advanced_image_processor with:
        - thought: "Applying Gaussian blur to smooth the image"
        - python_code: "blurred = cv2.GaussianBlur(image, (15, 15), 0); output_images.append(blurred)"
        - image_path: state['image_path']
        - operation_type: "filter"
        """
        try:
            messages = state["messages"]
            if not any(isinstance(msg, SystemMessage) for msg in messages):
                messages = [SystemMessage(content=system_prompt)] + messages
                
            if state.get("image_path"):
                for msg in messages:
                    if isinstance(msg, HumanMessage) and "image_path" not in msg.content:
                        msg.content += f"\nImage path: {state['image_path']}"
            
            response = llm_with_tools.invoke(messages)
            return {"messages": [response], "image_path": state.get("image_path", "")}
        except Exception as e:
            logger.error(f"LLM call error: {e}")
            return {
                "messages": [AIMessage(content=f"I encountered an error: {str(e)}. Please try again.")],
                "image_path": state.get("image_path", "")
            }

    builder = StateGraph(State)
    builder.add_node("tool_calling_llm", call_llm_model)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "tool_calling_llm")
    builder.add_conditional_edges("tool_calling_llm", tools_condition)
    builder.add_edge("tools", "tool_calling_llm")
    
    return builder.compile()