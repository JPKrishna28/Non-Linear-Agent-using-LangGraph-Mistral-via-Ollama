import streamlit as st
import pandas as pd
from main import create_graph, memory, visualize_graph
import json
from pathlib import Path
import time
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="LangGraph Router System",
    page_icon="🚀",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .output-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .status-box {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'app' not in st.session_state:
    st.session_state.app = create_graph()


def load_conversation_history():
    """Load conversation history from memory.json"""
    memory_file = Path("memory.json")
    if memory_file.exists():
        with open(memory_file, 'r') as f:
            data = json.load(f)
            return data.get("conversations", [])
    return []


def display_status(message, status_type="info"):
    """Display status messages with appropriate styling"""
    colors = {
        "info": "blue",
        "success": "green",
        "error": "red",
        "warning": "orange"
    }
    st.markdown(
        f"""<div class="status-box" style="background-color: {colors[status_type]}20;
        border-left: 5px solid {colors[status_type]}">
        {message}</div>""",
        unsafe_allow_html=True
    )


def main():
    # Header
    st.title("🚀 LangGraph Router System")
    st.markdown("---")

    # Sidebar - System Information and Settings
    with st.sidebar:
        st.header("System Information")
        st.markdown("---")

        # Display conversation count
        conversations = load_conversation_history()
        st.metric("Total Conversations", len(conversations))

        # Display system status
        st.subheader("System Status")
        st.success("Router: Online")
        st.success("Memory System: Active")
        st.success("LLM: Connected")

        # Display graph visualization
        st.subheader("System Architecture")
        visualize_graph("graph_structure.png")
        st.image("graph_structure.png", use_column_width=True)

    # Main content area - split into tabs
    tab1, tab2 = st.tabs(["Chat Interface", "Conversation History"])

    # Tab 1 - Chat Interface
    with tab1:
        # User input
        user_input = st.text_area("Enter your query:", height=100,
                                  placeholder="e.g., 'Calculate 15 * 3 and write a story about it'")

        if st.button("Process Query", type="primary"):
            if user_input:
                try:
                    # Show processing status
                    with st.spinner("Processing your query..."):
                        # Initialize state
                        initial_state = {
                            "input": user_input,
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

                        # Process the query
                        result = st.session_state.app.invoke(initial_state)

                        # Display results
                        st.markdown("### Processing Results")

                        # Show detected route
                        route_info = f"📍 Detected Route: **{result['route'].upper()}**"
                        if result['route'] == "multi":
                            route_info += f"\n\nSub-routes: {', '.join(result['sub_routes'])}"
                        st.markdown(route_info)

                        # Display memory context if available
                        if result.get('memory_context'):
                            with st.expander("🧠 Memory Context"):
                                st.markdown(result['memory_context'])

                        # Display final output
                        st.markdown("### Output")
                        st.markdown('<div class="output-box">', unsafe_allow_html=True)
                        st.markdown(result['final_output'])
                        st.markdown('</div>', unsafe_allow_html=True)

                        display_status("Query processed successfully!", "success")

                except Exception as e:
                    display_status(f"Error processing query: {str(e)}", "error")
            else:
                display_status("Please enter a query first!", "warning")

    # Tab 2 - Conversation History
    with tab2:
        conversations = load_conversation_history()

        if conversations:
            # Convert to DataFrame for better display
            df = pd.DataFrame(conversations)

            # Format timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

            # Create an expander for each conversation
            for idx, row in df.iloc[::-1].iterrows():
                with st.expander(f"🗣️ {row['query'][:50]}... ({row['timestamp']})"):
                    st.markdown(f"**Query:** {row['query']}")
                    st.markdown(f"**Route:** {row['route']}")
                    if row['metadata']:
                        st.markdown("**Processing Details:**")
                        for key, value in row['metadata'].items():
                            st.markdown(f"- {key}: {value}")
                    st.markdown("**Response:**")
                    st.markdown('<div class="output-box">', unsafe_allow_html=True)
                    st.markdown(row['response'])
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No conversation history available yet.")


if __name__ == "__main__":
    main()