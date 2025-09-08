from typing import List
from uuid import uuid4

from tqdm import tqdm

import pandas as pd
import numpy as np

from pydantic import BaseModel
from llama_index.llms.openai_like import OpenAILike


from llama_index.core.program import LLMTextCompletionProgram


llm = OpenAILike(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    api_base="https://3qocbzi14ao7bp-8000.proxy.runpod.net/v1",
    api_key="fake",
    context_window=8192,
    is_chat_model=True,
)



df = pd.read_csv("./data/doc_with_persona.csv")


with open("./question_gen_persona_prompt.md", "r") as f:
    p_prompt = f.read()
    
with open("./question_gen_general_prompt.md", "r") as f:
    g_prompt = f.read()
    
    
class Questions(BaseModel):
    q: List[str]
    
    
def run_program(prompt, documents, persona=None):
    program = LLMTextCompletionProgram.from_defaults(
        output_cls=Questions,
        prompt_template_str=prompt,
        llm=llm,
        verbose=True,
    )
    
    return program(documents=documents, persona=persona)

def gen_persona_q(persons, program_runner, documents, prompt):
    questions = []
    for p in persons:
        res = program_runner(prompt, documents, p)
        questions += res.q 
    return questions


data = []
error_docs = []
for _, row in tqdm(df.iterrows(), total=df.shape[0]):
    try: 
        questions = []
        if pd.isna(row["persona"]):
            res = run_program(g_prompt, row["doc"])
            questions += res.q 
        else: 
            questions =  gen_persona_q(row["persona"].split(","), run_program, row["doc"], p_prompt)
        
    
        for q in questions:
            data.append({
                "id": str(uuid4()),
                "doc_id": row["id"],
                "question": q
            })
    except :
        error_docs.append({"doc_id":row["id"]})
        
pd.DataFrame(data).to_csv("./data/questions.csv", index=False)
pd.DataFrame(error_docs).to_csv("./data/question_error.csv", index=False)
