import json
from harry_logic import epistemic_gate,ask_harry
from pathlib import Path

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

evaluate_gate(eval_text,facts_text)

