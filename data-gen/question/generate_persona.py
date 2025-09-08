import pandas as pd
from tqdm import tqdm
from typing import List, Literal

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.ollama import Ollama
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.llms.openai_like import OpenAILike


from pydantic import BaseModel, Field, conlist, confloat
from typing import List, Optional
from enum import Enum



llm = OpenAILike(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    api_base="https://3qocbzi14ao7bp-8000.proxy.runpod.net/v1",
    api_key="fake",
    context_window=8192,
    is_chat_model=True,
)


PersonaId = Literal[
    "employee","employer","hr_manager","domestic_worker","job_seeker",
    "business_owner","contractor","legal_advisor","government_official",
    "student","emirati","agency",
]

class PersonaItem(BaseModel):
    id: PersonaId                      
    score: float                       
    evidence: List[str] = Field(default_factory=list)

class PersonaPrediction(BaseModel):
    personas: List[PersonaItem] = Field(default_factory=list)
    
    

if __name__ == "__main__":    
    df = pd.read_csv("./data/doc_data.csv")

    with open("./persona_prompt.md", "r") as f: 
        prompt = f.read()
        
        
    program = LLMTextCompletionProgram.from_defaults(
        output_cls=PersonaPrediction,
        prompt_template_str=prompt,
        llm=llm,
        verbose=True,
    )
        
    persona_data = []
    error_data = []
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Generating"):
        try: 
            doc = row["doc"]
            id = row["id"]
            result = program(document_content=doc)
    
            personas = [p.id for p in result.personas if p.score >= 7 ]
            
            persona_data.append({
                "id": id, 
                "persona": ",".join(personas) if len(personas) != 0 else None
            })
        except: 
            error_data.append(idx)
        
    
    persona_data_df = pd.DataFrame(persona_data)
    pd.DataFrame({"id": error_data}).to_csv("./data/persona_error.csv", index=False)
    

    df.merge(persona_data_df, on="id").to_csv("./data/doc_with_persona.csv", index=False)
    
    