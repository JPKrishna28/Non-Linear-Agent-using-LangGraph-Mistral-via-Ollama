from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, START, END
from typing import Dict, Any, Literal, Optional, List
import re
import json
import logging
from datetime import datetime
from memory_manager import Memory
from graph_visualizer import visualize_graph
from logger_config import setup_logging

# Set up logging
logger = setup_logging()

# Initialize the LLM
llm = OllamaLLM(model="mistral")

# Global memory instance
memory = Memory()

def router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Route the query to appropriate processing nodes"""
    query = state['input'].lower()
    logger.info(f"Router processing: {query}")

    # Enhanced pattern matching for different operations
    math_patterns = [
        r'\b(calculate|compute|solve|find|what is)\b',
        r'\d+\s*[\+\-\*\/]\s*\d+',
        r'\bsquare root\b',
        r'\b(add|subtract|multiply|divide)\b',
        r'\b(sum|difference|product|quotient)\b'
    ]

    write_patterns = [
        r'\b(write|create|compose|tell)\b.*\b(story|tale|poem|essay|narrative)\b',
        r'\bwrite\b',
        r'\bstory\b',
        r'\bpoem\b',
        r'\bessay\b'
    ]

    translate_patterns = [
        r'\b(translate|translation|convert)\b',
        r'\bto\s+(spanish|french|german|italian|portuguese|chinese|japanese|korean|hindi|arabic)\b',
        r'\bin\s+(spanish|french|german|italian|portuguese|chinese|japanese|korean|hindi|arabic)\b'
    ]

    has_math = any(re.search(pattern, query) for pattern in math_patterns)
    has_write = any(re.search(pattern, query) for pattern in write_patterns)
    has_translate = any(re.search(pattern, query) for pattern in translate_patterns)

    # Determine route based on patterns
    routes = []
    if has_math:
        routes.append("math")
    if has_write:
        routes.append("write")
    if has_translate:
        routes.append("translate")

    if len(routes) > 1:
        state['route'] = "multi"
        state['sub_routes'] = routes
    elif len(routes) == 1:
        state['route'] = routes[0]
    else:
        state['route'] = "default"

    # Add context from memory
    context = memory.get_context(query)
    if context:
        state['memory_context'] = context

    logger.info(f"Router detected route: {state['route']}")
    return state

def math_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mathematical calculations"""
    query = state['input']
    logger.info(f"Math node processing: {query}")

    math_result = ""

    # Pattern matching for basic operations
    patterns = [
        (r'(\d+)\s*\+\s*(\d+)', lambda a, b: f"{a} + {b} = {int(a) + int(b)}"),
        (r'(\d+)\s*\-\s*(\d+)', lambda a, b: f"{a} - {b} = {int(a) - int(b)}"),
        (r'(\d+)\s*\*\s*(\d+)', lambda a, b: f"{a} * {b} = {int(a) * int(b)}"),
        (r'(\d+)\s*\/\s*(\d+)', lambda a, b: f"{a} / {b} = {float(a) / float(b)}" if int(b) != 0 else "Cannot divide by zero"),
        (r'square root of (\d+)', lambda a: f"Square root of {a} = {float(a) ** 0.5}")
    ]

    for pattern, calc_func in patterns:
        match = re.search(pattern, query)
        if match:
            try:
                if "square root" in pattern:
                    math_result = calc_func(float(match.group(1)))
                else:
                    math_result = calc_func(match.group(1), match.group(2))
                break
            except Exception as e:
                math_result = f"Error in calculation: {str(e)}"
                logger.error(f"Math calculation error: {e}")

    if not math_result:
        try:
            context = state.get('memory_context', '')
            prompt = f"""Previous context: {context}

Please solve this math problem and provide a clear answer: {query}"""
            math_result = llm.invoke(prompt)
        except Exception as e:
            math_result = f"Error calculating: {str(e)}"
            logger.error(f"LLM math error: {e}")

    state['math_result'] = math_result
    logger.info(f"Math result generated: {len(math_result)} characters")
    return state

def writer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Handle creative writing tasks"""
    query = state['input']
    logger.info(f"Writer node processing: {query}")

    math_context = state.get('math_result', '')
    memory_context = state.get('memory_context', '')

    try:
        if math_context:
            prompt = f"""Previous context: {memory_context}

Write a creative story that incorporates this math result: {math_context}

Original request: {query}

Please create an engaging story that naturally includes the mathematical calculation."""
        else:
            prompt = f"""Previous context: {memory_context}

Create engaging creative content based on this request: {query}"""

        story = llm.invoke(prompt)
        state['story'] = story
        logger.info(f"Story created: {len(story)} characters")

    except Exception as e:
        state['story'] = f"Error creating story: {str(e)}"
        logger.error(f"Writer error: {e}")

    return state

def translator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Handle translation tasks"""
    query = state['input']
    logger.info(f"Translator node processing: {query}")

    languages = {
        'spanish': 'Spanish', 'french': 'French', 'german': 'German',
        'italian': 'Italian', 'portuguese': 'Portuguese', 'chinese': 'Chinese',
        'japanese': 'Japanese', 'korean': 'Korean', 'hindi': 'Hindi', 'arabic': 'Arabic'
    }

    target_language = None
    for lang_key, lang_name in languages.items():
        if re.search(rf'\b(to|in)\s+{lang_key}\b', query.lower()):
            target_language = lang_name
            break

    if not target_language:
        target_language = "Spanish"  # Default

    try:
        content_to_translate = ""
        if state.get('math_result'):
            content_to_translate += f"Math: {state['math_result']}\n"
        if state.get('story'):
            content_to_translate += f"Story: {state['story']}\n"

        if not content_to_translate:
            translate_match = re.search(r'translate\s+"([^"]+)"', query)
            if translate_match:
                content_to_translate = translate_match.group(1)
            else:
                content_to_translate = query

        memory_context = state.get('memory_context', '')
        prompt = f"""Previous context: {memory_context}

Please translate the following content to {target_language}:

{content_to_translate}

Provide a natural, accurate translation."""

        translation = llm.invoke(prompt)
        state['translation'] = translation
        state['target_language'] = target_language
        logger.info(f"Translation to {target_language} completed: {len(translation)} characters")

    except Exception as e:
        state['translation'] = f"Error translating: {str(e)}"
        logger.error(f"Translation error: {e}")

    return state

def default_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Handle general queries"""
    query = state['input']
    logger.info(f"Default node processing: {query}")

    try:
        memory_context = state.get('memory_context', '')
        prompt = f"""Previous context: {memory_context}

Please provide a helpful response to: {query}"""

        response = llm.invoke(prompt)
        state['default_result'] = response
        logger.info(f"Default response generated: {len(response)} characters")

    except Exception as e:
        state['default_result'] = f"Error generating response: {str(e)}"
        logger.error(f"Default node error: {e}")

    return state

def final_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Combine results and generate final output"""
    route = state['route']
    logger.info(f"Final node processing route: {route}")

    output_parts = []

    if route == "multi":
        sub_routes = state.get('sub_routes', [])
        if 'math' in sub_routes and state.get('math_result'):
            output_parts.append(f"MATH CALCULATION:\n{state['math_result']}")
        if 'write' in sub_routes and state.get('story'):
            output_parts.append(f"CREATIVE STORY:\n{state['story']}")
        if 'translate' in sub_routes and state.get('translation'):
            target_lang = state.get('target_language', 'Unknown')
            output_parts.append(f"TRANSLATION ({target_lang}):\n{state['translation']}")
    elif route == "math":
        output_parts.append(state.get('math_result', 'No math result available'))
    elif route == "write":
        output_parts.append(state.get('story', 'No story available'))
    elif route == "translate":
        target_lang = state.get('target_language', 'Unknown')
        output_parts.append(f"Translation to {target_lang}:\n{state.get('translation', 'No translation available')}")
    else:
        output_parts.append(state.get('default_result', 'No response available'))

    final_output = "\n\n".join(output_parts)
    state['final_output'] = final_output

    # Save to memory
    memory.add_conversation(
        query=state['input'],
        response=final_output,
        route=route,
        metadata={
            'has_math': bool(state.get('math_result')),
            'has_story': bool(state.get('story')),
            'has_translation': bool(state.get('translation')),
            'target_language': state.get('target_language')
        }
    )

    logger.info(f"Final output generated: {len(final_output)} characters")
    return state

def create_router_condition(state: Dict[str, Any]) -> Literal[
    "math_node", "writer_node", "translator_node", "default_node"]:
    """Determine which node to route to from the router"""
    route = state['route']
    if route == "math":
        return "math_node"
    elif route == "write":
        return "writer_node"
    elif route == "translate":
        return "translator_node"
    elif route == "multi":
        sub_routes = state.get('sub_routes', [])
        if 'math' in sub_routes:
            return "math_node"
        elif 'write' in sub_routes:
            return "writer_node"
        elif 'translate' in sub_routes:
            return "translator_node"
    return "default_node"

def create_multi_condition(state: Dict[str, Any]) -> Literal["writer_node", "translator_node", "final_node"]:
    """Handle multi-route logic"""
    if state['route'] == "multi":
        sub_routes = state.get('sub_routes', [])
        if 'write' in sub_routes and not state.get('story'):
            return "writer_node"
        elif 'translate' in sub_routes and not state.get('translation'):
            return "translator_node"
    return "final_node"

def create_graph():
    """Create and configure the LangGraph"""
    logger.info("Creating LangGraph...")

    # Initialize the graph
    graph = StateGraph(dict)

    # Add all nodes
    graph.add_node("router", router_node)
    graph.add_node("math_node", math_node)
    graph.add_node("writer_node", writer_node)
    graph.add_node("translator_node", translator_node)
    graph.add_node("default_node", default_node)
    graph.add_node("final_node", final_node)

    # Set entry point
    graph.set_entry_point("router")

    # Add conditional edges from router
    graph.add_conditional_edges(
        "router",
        create_router_condition,
        {
            "math_node": "math_node",
            "writer_node": "writer_node",
            "translator_node": "translator_node",
            "default_node": "default_node"
        }
    )

    # Add conditional edges for multi-route processing
    graph.add_conditional_edges(
        "math_node",
        create_multi_condition,
        {
            "writer_node": "writer_node",
            "translator_node": "translator_node",
            "final_node": "final_node"
        }
    )

    graph.add_conditional_edges(
        "writer_node",
        create_multi_condition,
        {
            "translator_node": "translator_node",
            "final_node": "final_node"
        }
    )

    # Add direct edges to final_node
    graph.add_edge("translator_node", "final_node")
    graph.add_edge("default_node", "final_node")

    # Set finish point
    graph.set_finish_point("final_node")

    logger.info("LangGraph created successfully")
    return graph.compile()

def main():
    """Main function to test the enhanced router system"""
    print("🚀 Creating Enhanced LangGraph Router System...")
    logger.info("Starting enhanced router system")

    # Create graph visualization
    visualize_graph()

    app = create_graph()

    # Enhanced test cases
    test_queries = [
        "What is 5+5 and write a story about it.",
        "Calculate 15 * 3 and translate the result to Spanish",
        "Write a poem about summer and translate it to French",
    ]

    print(f"\n📊 Testing {len(test_queries)} queries...")

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 80}")
        print(f"🔍 TEST {i}: {query}")
        print(f"{'=' * 80}")

        try:
            # Initialize state
            initial_state = {
                "input": query,
                "route": "",
                "sub_routes": [],
                "math_result": "",
                "story": "",
                "translation": "",
                "target_language": "",
                "default_result": "",
                "memory_context": "",
                "final_output": ""
            }

            # Execute the graph
            result = app.invoke(initial_state)

            print(f"\n📝 FINAL OUTPUT:")
            print(f"{'-' * 60}")
            print(result['final_output'])

        except Exception as e:
            logger.error(f"Error processing query {i}: {e}")
            print(f"❌ Error processing query: {str(e)}")

        print(f"{'=' * 80}")

    print(f"\n💾 Memory contains {len(memory.conversations)} conversations")
    print(f"📊 Graph visualization saved as 'graph_structure.png'")
    print(f"📋 Logs saved to 'langgraph_router.log'")
    print(f"🧠 Memory saved to 'memory.json'")

if __name__ == "__main__":
    main()