I'll help you create a pull request to add detailed cloning and running instructions to your README.md file.I'll help you update the README with comprehensive installation and running instructions. Here's the updated content:

```markdown name=README.md
# LangGraph Router System

A sophisticated natural language processing system built with LangGraph that routes and processes queries through multiple specialized nodes including mathematical calculations, creative writing, and translation capabilities.

## Features

- Multi-route query processing
- Mathematical calculations
- Creative writing generation
- Language translation
- Memory system for conversation history
- Graph visualization
- Streamlit UI
- Comprehensive logging

## System Requirements

- Python 3.9 or higher
- 4GB RAM minimum (8GB recommended)
- Ollama installed and running locally
- Git for version control

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/JPKrishna28/Non-Linear-Agent-using-LangGraph-Mistral-via-Ollama.git
cd Non-Linear-Agent-using-LangGraph-Mistral-via-Ollama
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Ollama (Required for LLM)

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull the Mistral model:
```bash
ollama pull mistral
```

## Running the Application

### Method 1: Streamlit UI (Recommended)

1. Ensure Ollama is running in the background
2. Start the Streamlit interface:
```bash
streamlit run app.py
```
3. Open your browser and navigate to http://localhost:8501

### Method 2: Command Line Interface

1. Ensure Ollama is running in the background
2. Run the main script:
```bash
python app.py
```

## File Structure

```
langgraph_router/
├── main.py              # Core LangGraph implementation
├── streamlit_app.py     # Streamlit user interface
├── memory_manager.py    # Memory system implementation
├── graph_visualizer.py  # Graph visualization utilities
├── logger_config.py     # Logging configuration
├── requirements.txt     # Project dependencies
├── .gitignore          # Git ignore rules
└── README.md           # Project documentation
```

## Configuration

### Environment Variables (Optional)

Create a `.env` file in the root directory:

```env
OLLAMA_HOST=http://localhost:11434
LOG_LEVEL=INFO
```

### Logging

Logs are stored in the `logs` directory:
- Application logs: `logs/langgraph_router.log`
- System generated files:
  - `memory.json`: Conversation history
  - `graph_structure.png`: System architecture visualization

## Troubleshooting

1. **Ollama Connection Issues**
   - Ensure Ollama is running: `ollama serve`
   - Verify Mistral model is installed: `ollama list`

2. **Memory/Performance Issues**
   - Clear memory: Delete `memory.json`
   - Restart the application

3. **UI Not Loading**
   - Check if port 8501 is available
   - Ensure all dependencies are installed
   - Verify Python version compatibility

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details
```

This updated README provides:
1. Clear system requirements
2. Step-by-step installation instructions
3. Multiple methods for running the application
4. Configuration options
5. Troubleshooting guide
6. Detailed file structure
7. Environment setup instructions

Users can now easily:
- Clone and set up the project
- Install dependencies
- Configure the environment
- Run the application through UI or CLI
- Troubleshoot common issues

Just replace `<REPO_URL>` with your actual repository URL when you implement this README.