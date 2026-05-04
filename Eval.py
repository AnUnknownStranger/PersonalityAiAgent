import json
from harry_logic import epistemic_gate,ask_harry
from pathlib import Path
import os
from langchain_deepseek import ChatDeepSeek
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
import glob
from pathlib import Path
import time

def evaluate_gate(evaluation_data, facts, chat_history=[]):  
    results = {
        "passed": 0,
        "failed": 0,
        "failures": []
    }
    print(f"Starting Evaluation on {len(evaluation_data)} questions for epistemic gate...\n")
    for item in evaluation_data:
        question = item['query']
        expected_valid = (item['expected'] == 'VALID')
        actual_valid = epistemic_gate(question, chat_history, facts)
        if actual_valid == expected_valid:
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["failures"].append({
                "query": question,
                "expected": item['expected'],
                "actual": "VALID" if actual_valid else "INVALID"
            })
    total = results["passed"] + results["failed"]
    accuracy = (results["passed"] / total) * 100
    print(f"Total Questions: {total}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}\n")

    return results



ROOT = Path.cwd()
FACTS_PATH = ROOT / 'Data_HP' / 'facts_harry.json'
DIALOGUES_PATH = ROOT / 'Data_HP' / 'Harry_all_clean.txt'
FILE_PATH = ROOT / 'Eval' / 'Epistemic_Gate.json'
with FACTS_PATH.open('r', encoding='utf-8') as f:
    facts_data = json.load(f)

facts_text = json.dumps(facts_data, ensure_ascii=False, indent=2)

with FILE_PATH.open('r', encoding='utf-8') as f:
    eval_data = json.load(f)

eval_text = questions_list = eval_data.get('evaluation_questions', eval_data)

#evaluate_gate(eval_text,facts_text)


llm = ChatDeepSeek(
    model="deepseek-chat", 
    temperature=0, 
    api_key="sk-da6420ca5a6748e9907c500c66adb470" 
)

JUDGE_SYSTEM_PROMPT = """
You are a Quality Assurance Judge for the Harry Potter Persona Engine. 
You are evaluating the 'Vocal Filter' gate.

### YOUR TASK:
Compare the response against the 'EXPECTED_STATUS'.
- If EXPECTED is 'VALID': The response should be in-character, British, and helpful.
- If EXPECTED is 'INVALID': The response MUST be a dismissal (e.g., calling it Muggle nonsense).

### RUBRIC:
1. TARGET ALIGNMENT: Did the response correctly handle the query based on the expected status?
2. VOICE (1-5): 5 = Blunt, unpolished, British. 0 = Sounds like a helpful AI assistant.
3. BREVITY: Fails if an 'INVALID' dismissal is longer than 2 sentences.

OUTPUT ONLY VALID JSON:
{
    "passed": bool,
    "voice_score": int,
    "alignment_score": int,
    "reasoning": "string"
}
"""


def judge(test_suite_path):
    with open(test_suite_path, 'r') as f:
        suite = json.load(f)
    report = []
    pass_count = 0
    t = []
    for item in suite['evaluation_questions']:
        st = time.perf_counter()
        harry_output = ask_harry(item['query'])
        et = time.perf_counter()
        dur = et-st
        t.append(dur)
        judge_input = f"""
        QUERY: {item['query']}
        EXPECTED_STATUS: {item['expected']}
        HARRY'S RESPONSE: {harry_output['response']}
        INTERNAL MOTIVE: {harry_output['reasoning']['motive']}
        """

        try:
            messages = [
                SystemMessage(content=JUDGE_SYSTEM_PROMPT),
                HumanMessage(content=judge_input),
            ]
            response = llm.invoke(messages)
            raw_content = response.content.replace('```json', '').replace('```', '').strip()
            assessment = json.loads(raw_content)
            
        except Exception as e:
            assessment = {"passed": False, "reasoning": f"Parsing Error: {str(e)}"}
        is_passed = assessment.get('passed') is True
        
        if is_passed:
            pass_count += 1

        report.append({
            "id": item['id'],
            "passed": is_passed,
            "scores": assessment
        })

    total = len(suite['evaluation_questions'])
    avg_latency = sum(t) / total
    print(f"\n--- VOCAL FILTER EVALUATION COMPLETE ---")
    print(f"Final Pass Rate: {(pass_count/total)*100:.2f}% ({pass_count}/{total})")
    print(f"Expected wait time per response: {avg_latency:.4f} seconds")
    return report


judge("Eval/VocalCheck.json")

