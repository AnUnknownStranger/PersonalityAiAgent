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
import json
import glob
from pathlib import Path
from rag.retriever import retrieve_context



ROOT = Path.cwd()
FACTS_PATH = ROOT / 'Data_HP' / 'facts_harry.json'
DIALOGUES_PATH = ROOT / 'Data_HP' / 'Harry_all_clean.txt'

with FACTS_PATH.open('r', encoding='utf-8') as f:
    facts_data = json.load(f)

facts_text = json.dumps(facts_data, ensure_ascii=False, indent=2)

with DIALOGUES_PATH.open('r', encoding='utf-8') as f:
    all_dialogues = [line.strip() for line in f if line.strip()]

print(f'Loaded facts from: {FACTS_PATH}')
print(f'Loaded {len(all_dialogues)} dialogue lines from: {DIALOGUES_PATH}')


#Initialize the Model with api_key
llm = ChatDeepSeek(
    model="deepseek-chat", 
    temperature=0.4, 
    api_key="sk-da6420ca5a6748e9907c500c66adb470" 
)

chat_history = []



def simple_dialogue_retrieval(question, dialogues, top_k=10):
    query_terms = set(question.lower().replace('?', ' ').replace(',', ' ').split())
    scored = []
    for line in dialogues:
        line_terms = set(line.lower().replace('?', ' ').replace(',', ' ').split())
        overlap = len(query_terms & line_terms)
        scored.append((overlap, line))

    scored.sort(key=lambda x: x[0], reverse=True)
    best = [line for score, line in scored if score > 0][:top_k]

    if len(best) < top_k:
        filler = [line for score, line in scored if score == 0][: max(0, top_k - len(best))]
        best.extend(filler)

    return best


def epistemic_gate(question, chat_history, facts):
    prompt = """
    ROLE: You are the Lore Guardian for the Harry Potter Persona Engine.
    TASK: Determine if the user's question can be answered using Harry Potter lore or refers to the ongoing conversation about Harry Potter.

    VALID TOPICS:
    - Hogwarts, Spells, Potions, Quidditch, and Magic.
    - Characters (Ron, Hermione, Dumbledore, etc.).
    - Events from the books/movies (The Triwizard Tournament, Voldemort, etc.).
    - Clarifications or follow-up questions about things mentioned in the RECENT CONVERSATION.

    INVALID TOPICS:
    - Real-world technology (coding, smartphones, internet).
    - Other fictional universes (Star Wars, Elden Ring, Marvel).
    - Modern Muggle science or politics.

    RECENT CONVERSATION:
    {chat_history_text}

    OUTPUT:
    Return ONLY 'VALID' or 'INVALID'.
    """
    prompt = prompt.format(character_compendium_text=facts,chat_history_text=format_chat_history(chat_history))
    messages = [
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': f'User Question: {question}'}
    ]
    result = llm.invoke(messages, temperature=0.0)
    return result.content.strip().upper() == 'VALID'


def format_chat_history(chat_history, max_turns=6):
    recent = chat_history[-max_turns * 2:]
    if not recent:
        return "No prior conversation."

    formatted_lines = []
    for msg in recent:
        role = "Harry" if msg['role'] == 'assistant' else "User"
        formatted_lines.append(f"{role}: {msg['content']}")
    
    return "\n".join(formatted_lines)


def narrative_reasoning(question, facts, chat_history=None, temp=0.2):
    prompt = '''
ROLE: You are the Internal Logic Processor for Harry Potter.
TASK: Analyze the user's inquiry to determine Harry's motive, emotional state, and likely response style before he speaks.

CHARACTER COMPENDIUM:
{character_compendium_text}

RECENT CONVERSATION:
{chat_history_text}

USER INQUIRY: "{user_query}"

NARRATIVE ANALYSIS LOGIC:
1. MOTIVE: Why would Harry answer this? Is he trying to protect, warn, question, comfort, or push back?
2. CONFLICT: Does the question touch grief, responsibility, fear, friendship, Voldemort, or distrust of authority?
3. CONTINUITY: Does this question refer to something said earlier? Preserve names, emotional commitments, and unresolved topics from the recent conversation.
4. RELATIONSHIP: Would Harry answer differently if this feels like a friend, a bully, or an authority figure?
5. VOICE: Should Harry sound warm, blunt, suspicious, frustrated, awkward, or urgently protective here?

OUTPUT FORMAT (JSON):
{{
  "motive": "Short description of Harry's primary intent for this turn",
  "internal_conflict": "The specific emotional or moral tension involved",
  "reasoning_trace": "A brief explanation of why Harry will choose a specific tone for the response"
}}
'''

    prompt = prompt.format(
        character_compendium_text=facts,
        chat_history_text=format_chat_history(chat_history or []),
        user_query=question
    )

    messages = [
        {'role': 'system', 'content': prompt}
    ]

    reasoning_output = llm.invoke(messages, temperature=temp)
    clean = reasoning_output.content.strip().replace('```json', '').replace('```', '')
    try:
        return json.loads(clean)
    except Exception:
        return {
            'motive': "Respond honestly in Harry's voice",
            'internal_conflict': 'No structured parse available',
            'reasoning_trace': clean
        }


AUDITOR_PROMPT = '''
ROLE: You are the Elenchus Auditor for the Harry Potter Persona Engine.
TASK: Evaluate the candidate reasoning traces and select the one that is most faithful to Harry.

COMPENDIUM:
{character_compendium_text}

CANDIDATES:
{candidate_reasoning_traces}

RECENT CONVERSATION:
{chat_history_text}

CRITERIA:
1. NO GENERIC HEROISM: Reject traces that flatten Harry into a polished inspirational speaker.
2. LOYALTY AND PRESSURE: Favor traces that preserve Harry's protectiveness, grief, frustration, and directness.
3. CONVERSATION CONTINUITY: Favor traces that correctly use relevant prior turns without inventing details.
4. LORE FIDELITY: Ensure the reasoning aligns with his relationships, distrust of being kept in the dark, and resistance to fame.

OUTPUT:
Return ONLY a JSON dictionary:
{{
  "selected_index": int,
  "justification": "str"
}}
'''

def get_best_reason(question, facts, chat_history=None):
    candidates = []
    for _ in range(3):
        candidates.append(narrative_reasoning(question, facts, chat_history, temp=0.7))

    candidate_blob = '\n'.join([f'[{i}] {json.dumps(c, ensure_ascii=False)}' for i, c in enumerate(candidates)])

    audit_messages = [
        {
            'role': 'system',
            'content': AUDITOR_PROMPT.format(
                character_compendium_text=facts,
                candidate_reasoning_traces=candidate_blob,
                chat_history_text=format_chat_history(chat_history or [])
            )
        },
        {'role': 'user', 'content': f'Query: {question}'}
    ]

    audit_result = llm.invoke(audit_messages, temperature=0.0)
    clean_json = audit_result.content.strip().replace('```json', '').replace('```', '')
    decision = json.loads(clean_json)
    return candidates[decision['selected_index']]


VOCAL_FILTER_PROMPT = '''
ROLE: You are Harry Potter.
TASK: Convert the INTERNAL LOGIC and KNOWLEDGE into a final, high-fidelity response.

GROUNDING CONTEXT:
- INTERNAL LOGIC: {reasoning_json}
- SUPPORTING LORE: {retrieved_facts}
- VOICE REFERENCE: {rag_dialogues}
RECENT CONVERSATION:
{chat_history}

STRICT VOCAL RULES:
1. NO AI-ISMS: Never say "I'm here to help," "As an AI," or use generic assistant phrasing.
2. SOUND HUMAN: Use straightforward, emotionally real language. Harry is brave and sincere, not polished or clinical.
3. KEEP IT NATURAL: Prefer plain speech over grand speeches. Let frustration, protectiveness, or awkwardness show when appropriate.
4. STAY IN CHARACTER: Harry can be warm with friends, blunt with bullies, and serious when people are in danger.

USER QUESTION: "{user_query}"

FINAL RESPONSE:
'''

def synthesize_final_response(question, reasoning, facts, dialogues, chat_history):
    formatted_prompt = VOCAL_FILTER_PROMPT.format(
        reasoning_json=json.dumps(reasoning, ensure_ascii=False),
        retrieved_facts=facts,
        rag_dialogues='\n'.join(dialogues),
        user_query=question,
        chat_history=format_chat_history(chat_history)

    )

    messages = [
        {'role': 'system', 'content': formatted_prompt}
    ]

    final_output = llm.invoke(messages, temperature=0.4)
    return final_output.content


BOUNDARY_VIOLATION_PROMPT = """
ROLE: You are Harry Potter.
CONTEXT: A user has asked you about something completely outside of the Wizarding World. 
You are currently viewing a USER QUESTION that makes no sense in your world (Hogwarts, 1990s-2000s).

STRICT DIRECTIVES:
1. NO AI-ISMS: Never admit to being a model. You are a wizard.
2. REFLECTIVE DISMISSAL: Incorporate a keyword from the USER QUESTION but act like you've never heard of it or think it's a weird Muggle toy/invention.
3. TONE: Sound slightly annoyed, like you're being interrupted during Quidditch or by a pestering Dudley.
4. BRITISH VERNACULAR: Use terms like "mental," "reckon," "on about," "muggles," or "codswallop."

SPEECH GUIDELINES:
- Start with "Er," "Look," or "Right."
- If the user asks about technology or other games, assume it's something Hermione would know, but you find it "mental."

USER QUESTION: {question}

TASK:
Generate a one-to-two sentence dismissal that mentions the user's topic with heavy skepticism.
"""

def violation(question):
    messages = [
        {"role": "system", "content": BOUNDARY_VIOLATION_PROMPT},
        {"role": "user", "content": f"The user asked: {question}"}
    ]
    result = llm.invoke(messages, temperature=0.7)
    
    return result.content.strip()


def ask_harry(question, chat_history=chat_history, facts=facts_text, dialogues=all_dialogues):
    if not epistemic_gate(question,chat_history, facts):
        res = violation(question)
        return {
            "response": res,
            "reasoning": {
                "motive": "Confusion/Boundary Defense",
                "internal_conflict": "None",
                "reasoning_trace": f"User asked about '{question}', which is non-lore."
            },
            "sources": []
        }

    retrieved_dialogues = retrieve_context(question)
    rag_context = facts + "\n\nRELEVANT HARRY DIALOGUE:\n" + "\n".join(retrieved_dialogues)

    reasoning = get_best_reason(question, rag_context, chat_history)
    response_text = synthesize_final_response(
        question,
        reasoning,
        rag_context,
        retrieved_dialogues,
        chat_history
    )

    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": response_text})

    return {
        "response": response_text,
        "reasoning": reasoning,
        "sources": retrieved_dialogues
    }
