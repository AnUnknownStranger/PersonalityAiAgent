#Harry Potter Persona System
## System Structure
-**Frontend**: A simple chat made with Streamlit
-**RAG**: Retrieve the relevant dialogue
-**LLM**:Deepseek API
### MultiAgent Pipeline
-**Epistemic Gate**: Filters out irrelevant knowledges
-**Narrative Reasoning**: Analyze the character’s motivation, internal conflict, and reasoning
-**Reasoning Selection**: Select the best reason
-**Vocal Filter**: Generates the final result based on the reasoning, dialogues, and supporting facts

## Setup instructions
```bash
pip install -r requirements.txt
```
## Execution
To start Chatting Run:
```bash
streamlit run server.py
```
To perform evaluations Run:
```bash
python Eval.py
```
## Example usage

Simply type 'what is Neural Network' to see the effect of epistemic gate

Start standard conversation with questions like "Hey Harry, want to go grab a Butterbeer?"

